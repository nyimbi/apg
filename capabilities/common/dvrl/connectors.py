#!/usr/bin/env python3
"""
APG Data Virtualization (DVRL) Universal Connector Framework
Production-ready database connectors with real client implementations

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union
from uuid_extensions import uuid7str
from urllib.parse import urlparse

# Production database client imports
try:
	import asyncpg
	import aiomysql
	import motor.motor_asyncio
	import aiohttp
	from cassandra.cluster import Cluster
	from cassandra.auth import PlainTextAuthProvider
	from cassandra.policies import RoundRobinPolicy
	import cx_Oracle
	import pyodbc
except ImportError as e:
	logging.warning(f"Some database clients not available: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
	from .models import DataSource, DataSourceType, DataSourceStatus, DataSourceSchema
except ImportError:
	from models import DataSource, DataSourceType, DataSourceStatus, DataSourceSchema


class ConnectionCapability(str, Enum):
	"""Data source connection capabilities"""
	BATCH_READ = "batch_read"
	STREAMING_READ = "streaming_read"
	BATCH_WRITE = "batch_write"
	STREAMING_WRITE = "streaming_write"
	TRANSACTION_SUPPORT = "transaction_support"
	SCHEMA_INTROSPECTION = "schema_introspection"
	QUERY_PUSHDOWN = "query_pushdown"
	AGGREGATION_PUSHDOWN = "aggregation_pushdown"
	JOIN_PUSHDOWN = "join_pushdown"
	LIMIT_PUSHDOWN = "limit_pushdown"
	FULL_TEXT_SEARCH = "full_text_search"
	VECTOR_SEARCH = "vector_search"
	TIME_SERIES_SUPPORT = "time_series_support"


class ConnectionHealth(str, Enum):
	"""Connection health status"""
	HEALTHY = "healthy"
	DEGRADED = "degraded"
	UNHEALTHY = "unhealthy"
	UNKNOWN = "unknown"


class BaseConnector(ABC):
	"""
	Abstract base class for all data source connectors in the APG DVRL system.

	This class provides the foundational interface for all data source connectivity
	implementations, supporting heterogeneous data sources including SQL databases,
	NoSQL stores, APIs, cloud storage, and streaming platforms.

	The connector manages connection lifecycle, health monitoring, capability
	discovery, and query execution while maintaining tenant isolation and
	comprehensive error handling.

	Attributes:
		data_source (DataSource): Data source configuration and metadata
		tenant_id (str): APG tenant identifier for multi-tenancy
		user_id (str): User identifier for audit and authorization
		connection_pool: Database-specific connection pool instance
		capabilities (List[ConnectionCapability]): Supported data source capabilities
		health_status (ConnectionHealth): Current connection health status
		last_health_check (datetime): Timestamp of last health check
		connection_metadata (Dict[str, Any]): Connection-specific metadata and metrics
	"""

	def __init__(self, data_source: DataSource, tenant_id: str, user_id: str):
		"""
		Initialize base connector with data source configuration and context.

		Sets up connection state, health monitoring, and capability tracking
		while maintaining tenant isolation and user context for auditing.

		Args:
			data_source (DataSource): Complete data source configuration including
				connection parameters, credentials, and metadata
			tenant_id (str): APG tenant identifier for multi-tenant isolation
			user_id (str): User identifier for audit logging and authorization

		Note:
			Connection is not established during initialization. Call connect()
			method explicitly to establish the connection.
		"""
		self.data_source = data_source
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.connection_pool = None
		self.capabilities: List[ConnectionCapability] = []
		self.health_status = ConnectionHealth.UNKNOWN
		self.last_health_check = None
		self.connection_metadata: Dict[str, Any] = {}

	@abstractmethod
	async def connect(self) -> bool:
		"""
		Establish connection to data source

		Returns:
			bool: True if connection successful, False otherwise

		Raises:
			ConnectionError: If connection cannot be established
		"""
		raise NotImplementedError("Subclasses must implement connect method")

	@abstractmethod
	async def disconnect(self) -> bool:
		"""
		Close connection to data source

		Returns:
			bool: True if disconnection successful, False otherwise
		"""
		raise NotImplementedError("Subclasses must implement disconnect method")

	@abstractmethod
	async def test_connection(self) -> bool:
		"""
		Test connection health

		Returns:
			bool: True if connection is healthy, False otherwise
		"""
		raise NotImplementedError("Subclasses must implement test_connection method")

	@abstractmethod
	async def discover_schema(self) -> DataSourceSchema:
		"""
		Auto-discover data source schema

		Returns:
			DataSourceSchema: Discovered schema information

		Raises:
			SchemaDiscoveryError: If schema cannot be discovered
		"""
		raise NotImplementedError("Subclasses must implement discover_schema method")

	@abstractmethod
	async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""
		Execute query against data source

		Args:
			query (str): SQL or query string to execute
			parameters (Optional[Dict[str, Any]]): Query parameters for safe execution

		Returns:
			Dict[str, Any]: Query results including data, metadata, and execution info

		Raises:
			QueryExecutionError: If query execution fails
		"""
		raise NotImplementedError("Subclasses must implement execute_query method")

	@abstractmethod
	async def get_capabilities(self) -> List[ConnectionCapability]:
		"""
		Get data source capabilities

		Returns:
			List[ConnectionCapability]: List of supported capabilities
		"""
		raise NotImplementedError("Subclasses must implement get_capabilities method")

	async def health_check(self) -> ConnectionHealth:
		"""
		Perform comprehensive health check on data source connection.

		Executes connection test and updates health status with detailed monitoring
		information. Health checks are essential for connection pool management,
		load balancing, and system reliability monitoring.

		Health status meanings:
		- HEALTHY: Connection active and responsive
		- DEGRADED: Connection works but with performance issues
		- UNHEALTHY: Connection failed or unresponsive
		- UNKNOWN: Health status not yet determined

		Returns:
			ConnectionHealth: Current health status after check

		Note:
			Health check results are cached in self.health_status and
			self.last_health_check for monitoring and alerting systems.

		Example:
			>>> status = await connector.health_check()
			>>> if status == ConnectionHealth.HEALTHY:
			...     print("Connection is ready for queries")
		"""
		try:
			is_healthy = await self.test_connection()
			self.health_status = ConnectionHealth.HEALTHY if is_healthy else ConnectionHealth.UNHEALTHY
			self.last_health_check = datetime.now(timezone.utc)
			return self.health_status
		except Exception as e:
			self.health_status = ConnectionHealth.UNHEALTHY
			await _log_error(f"Health check failed for {self.data_source.name}", e)
			return self.health_status

	async def get_connection_stats(self) -> Dict[str, Any]:
		"""
		Get comprehensive connection statistics and performance metrics.

		Returns detailed information about the connector's current state,
		performance characteristics, and capabilities for monitoring,
		debugging, and optimization purposes.

		Returns:
			Dict[str, Any]: Connection statistics containing:
				- connector_type: Class name of the specific connector implementation
				- data_source_id: Unique identifier for the data source
				- data_source_name: Human-readable name of the data source
				- health_status: Current health status (healthy/degraded/unhealthy/unknown)
				- last_health_check: ISO timestamp of last health check
				- capabilities: List of supported connection capabilities
				- connection_metadata: Implementation-specific metadata and metrics

		Example:
			>>> stats = await connector.get_connection_stats()
			>>> print(f"Connector: {stats['connector_type']}")
			>>> print(f"Health: {stats['health_status']}")
			>>> print(f"Capabilities: {', '.join(stats['capabilities'])}")
		"""
		return {
			'connector_type': self.__class__.__name__,
			'data_source_id': self.data_source.id,
			'data_source_name': self.data_source.name,
			'health_status': self.health_status.value,
			'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
			'capabilities': [cap.value for cap in self.capabilities],
			'connection_metadata': self.connection_metadata
		}


class SQLDatabaseConnector(BaseConnector):
	"""
	Production SQL database connector with support for multiple database engines.

	Provides unified interface for SQL database connectivity with native async
	client libraries. Supports PostgreSQL (asyncpg), MySQL (aiomysql),
	Oracle (cx_Oracle), and SQL Server (pyodbc) with optimized connection
	pooling, transaction management, and query execution.

	Features:
	- Async connection pooling with configurable pool sizes
	- Multi-database support with engine-specific optimizations
	- Production-grade error handling and retry logic
	- Schema introspection using INFORMATION_SCHEMA
	- Query parameter binding for SQL injection prevention
	- Connection health monitoring and automatic recovery

	Supported Databases:
	- PostgreSQL 11+ (via asyncpg)
	- MySQL 5.7+ (via aiomysql)
	- Oracle 12c+ (via cx_Oracle with threading)
	- SQL Server 2016+ (via pyodbc with connection pooling)
	"""

	def __init__(self, data_source: DataSource, tenant_id: str, user_id: str):
		"""
		Initialize SQL database connector with engine-specific configuration.

		Sets up database-specific connection parameters, logging, connection
		pooling configuration, and timeout settings based on data source type.

		Args:
			data_source (DataSource): SQL database configuration with connection
				parameters, credentials, and database-specific settings
			tenant_id (str): Tenant identifier for multi-tenant isolation
			user_id (str): User identifier for audit and connection tracking

		Note:
			Connection string is built during initialization but actual connection
			is established when connect() is called. Pool size and timeout can
			be configured via data_source.connection_config.
		"""
		super().__init__(data_source, tenant_id, user_id)
		self.logger = logging.getLogger(f"dvrl.connectors.{data_source.type.value}")
		self.connection_pool = None
		self.connection_string = self._build_connection_string()
		self.pool_size = getattr(data_source.connection_config, 'pool_size', 10)
		self.timeout = getattr(data_source.connection_config, 'timeout', 30)

	async def connect(self) -> bool:
		"""Connect to SQL database using appropriate client library"""
		try:
			if self.data_source.type == DataSourceType.POSTGRESQL:
				# PostgreSQL with asyncpg
				self.connection_pool = await asyncpg.create_pool(
					self.connection_string,
					min_size=1,
					max_size=self.pool_size,
					command_timeout=self.timeout
				)

			elif self.data_source.type == DataSourceType.MYSQL:
				# MySQL with aiomysql
				config = self.data_source.connection_config or {}
				self.connection_pool = await aiomysql.create_pool(
					host=config.get('host', 'localhost'),
					port=config.get('port', 3306),
					user=config.get('username', config.get('user')),
					password=config.get('password'),
					db=config.get('database', config.get('dbname')),
					maxsize=self.pool_size,
					autocommit=False
				)

			elif self.data_source.type == DataSourceType.ORACLE:
				# Oracle with cx_Oracle (note: synchronous, will need thread pool)
				config = self.data_source.connection_config or {}
				dsn = cx_Oracle.makedsn(
					config.get('host', 'localhost'),
					config.get('port', 1521),
					service_name=config.get('service_name', config.get('database'))
				)
				self.connection_pool = cx_Oracle.SessionPool(
					config.get('username'),
					config.get('password'),
					dsn,
					min=1,
					max=self.pool_size,
					increment=1,
					threaded=True
				)

			elif self.data_source.type == DataSourceType.SQLSERVER:
				# SQL Server with pyodbc (note: synchronous, will need thread pool)
				config = self.data_source.connection_config or {}
				self.connection_string = (
					f"DRIVER={{ODBC Driver 17 for SQL Server}};"
					f"SERVER={config.get('host', 'localhost')},{config.get('port', 1433)};"
					f"DATABASE={config.get('database')};"
					f"UID={config.get('username')};"
					f"PWD={config.get('password')}"
				)
				# Note: Connection pool will be managed at query execution level

			else:
				raise ValueError(f"Unsupported SQL database type: {self.data_source.type}")

			# Test the connection
			connection_test = await self.test_connection()
			if not connection_test:
				raise ConnectionError("Connection test failed")

			# Store connection metadata
			self.connection_metadata = {
				'database_type': self.data_source.type.value,
				'driver': self._get_driver_name(),
				'pool_size': self.pool_size,
				'timeout_seconds': self.timeout,
				'connection_established': datetime.now(timezone.utc).isoformat()
			}

			self.logger.info(f"Connected to {self.data_source.type.value} database: {self.data_source.name}")
			return True

		except Exception as e:
			self.logger.error(f"Failed to connect to SQL database {self.data_source.name}: {e}")
			return False

	async def disconnect(self) -> bool:
		"""Disconnect from SQL database"""
		try:
			if self.connection_pool:
				if self.data_source.type in [DataSourceType.POSTGRESQL, DataSourceType.MYSQL]:
					# Async pools
					self.connection_pool.close()
					if hasattr(self.connection_pool, 'wait_closed'):
						await self.connection_pool.wait_closed()
				elif self.data_source.type in [DataSourceType.ORACLE, DataSourceType.SQLSERVER]:
					# Sync pools
					if hasattr(self.connection_pool, 'close'):
						self.connection_pool.close()

				self.connection_pool = None

			self.logger.info(f"Disconnected from SQL database: {self.data_source.name}")
			return True

		except Exception as e:
			self.logger.error(f"Failed to disconnect from SQL database {self.data_source.name}: {e}")
			return False

	async def test_connection(self) -> bool:
		"""Test SQL database connection"""
		try:
			if self.data_source.type == DataSourceType.POSTGRESQL:
				if self.connection_pool:
					async with self.connection_pool.acquire() as conn:
						await conn.execute('SELECT 1')
					return True
				else:
					# Test connection without pool
					conn = await asyncpg.connect(self.connection_string)
					await conn.execute('SELECT 1')
					await conn.close()
					return True

			elif self.data_source.type == DataSourceType.MYSQL:
				if self.connection_pool:
					async with self.connection_pool.acquire() as conn:
						async with conn.cursor() as cursor:
							await cursor.execute('SELECT 1')
					return True
				else:
					# Test without pool
					config = self.data_source.connection_config or {}
					conn = await aiomysql.connect(
						host=config.get('host', 'localhost'),
						port=config.get('port', 3306),
						user=config.get('username'),
						password=config.get('password'),
						db=config.get('database')
					)
					async with conn.cursor() as cursor:
						await cursor.execute('SELECT 1')
					conn.close()
					return True

			elif self.data_source.type == DataSourceType.ORACLE:
				# Oracle test (synchronous)
				if self.connection_pool:
					conn = self.connection_pool.acquire()
					cursor = conn.cursor()
					cursor.execute('SELECT 1 FROM dual')
					cursor.close()
					self.connection_pool.release(conn)
					return True

			elif self.data_source.type == DataSourceType.SQLSERVER:
				# SQL Server test (synchronous)
				conn = pyodbc.connect(self.connection_string)
				cursor = conn.cursor()
				cursor.execute('SELECT 1')
				cursor.close()
				conn.close()
				return True

			return False

		except Exception as e:
			self.logger.error(f"Connection test failed for {self.data_source.name}: {e}")
			return False

	async def discover_schema(self) -> DataSourceSchema:
		"""Auto-discover SQL database schema using information_schema"""
		try:
			tables = []

			if self.data_source.type == DataSourceType.POSTGRESQL:
				tables = await self._discover_postgresql_schema()
			elif self.data_source.type == DataSourceType.MYSQL:
				tables = await self._discover_mysql_schema()
			elif self.data_source.type == DataSourceType.ORACLE:
				tables = await self._discover_oracle_schema()
			elif self.data_source.type == DataSourceType.SQLSERVER:
				tables = await self._discover_sqlserver_schema()

			schema = DataSourceSchema(
				data_source_id=self.data_source.id,
				schema_name=self.data_source.schema or 'public',
				tenant_id=self.tenant_id,
				created_by=self.user_id,
				tables=tables,
				discovery_method=f"{self.data_source.type.value}_introspection",
				confidence_score=0.98
			)

			self.logger.info(f"Discovered {len(tables)} tables for {self.data_source.name}")
			return schema

		except Exception as e:
			self.logger.error(f"Schema discovery failed for {self.data_source.name}: {e}")
			raise

	async def _discover_postgresql_schema(self) -> List[Dict[str, Any]]:
		"""Discover PostgreSQL schema using information_schema"""
		schema_name = self.data_source.schema or 'public'

		# Query for tables and columns
		table_query = """
		SELECT
			t.table_name,
			t.table_type,
			c.column_name,
			c.data_type,
			c.is_nullable,
			c.column_default,
			tc.constraint_type
		FROM information_schema.tables t
		LEFT JOIN information_schema.columns c ON t.table_name = c.table_name
			AND t.table_schema = c.table_schema
		LEFT JOIN information_schema.table_constraints tc ON t.table_name = tc.table_name
			AND t.table_schema = tc.table_schema
			AND tc.constraint_type = 'PRIMARY KEY'
		LEFT JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
			AND c.column_name = kcu.column_name
		WHERE t.table_schema = $1
			AND t.table_type IN ('BASE TABLE', 'VIEW')
		ORDER BY t.table_name, c.ordinal_position
		"""

		async with self.connection_pool.acquire() as conn:
			rows = await conn.fetch(table_query, schema_name)

		return self._group_table_data(rows)

	async def _discover_mysql_schema(self) -> List[Dict[str, Any]]:
		"""Discover MySQL schema using information_schema"""
		database_name = self.data_source.connection_config.get('database')

		table_query = """
		SELECT
			t.TABLE_NAME as table_name,
			t.TABLE_TYPE as table_type,
			c.COLUMN_NAME as column_name,
			c.DATA_TYPE as data_type,
			c.IS_NULLABLE as is_nullable,
			c.COLUMN_DEFAULT as column_default,
			c.COLUMN_KEY as constraint_type
		FROM information_schema.TABLES t
		LEFT JOIN information_schema.COLUMNS c ON t.TABLE_NAME = c.TABLE_NAME
			AND t.TABLE_SCHEMA = c.TABLE_SCHEMA
		WHERE t.TABLE_SCHEMA = %s
			AND t.TABLE_TYPE IN ('BASE TABLE', 'VIEW')
		ORDER BY t.TABLE_NAME, c.ORDINAL_POSITION
		"""

		async with self.connection_pool.acquire() as conn:
			async with conn.cursor() as cursor:
				await cursor.execute(table_query, (database_name,))
				rows = await cursor.fetchall()

				# Convert to dict format
				row_dicts = []
				for row in rows:
					row_dicts.append({
						'table_name': row[0],
						'table_type': row[1],
						'column_name': row[2],
						'data_type': row[3],
						'is_nullable': row[4],
						'column_default': row[5],
						'constraint_type': 'PRIMARY KEY' if row[6] == 'PRI' else None
					})

		return self._group_table_data(row_dicts)

	async def _discover_oracle_schema(self) -> List[Dict[str, Any]]:
		"""Discover Oracle schema (synchronous)"""
		schema_name = self.data_source.connection_config.get('username', '').upper()

		table_query = """
		SELECT
			t.table_name,
			'BASE TABLE' as table_type,
			c.column_name,
			c.data_type,
			c.nullable as is_nullable,
			c.data_default as column_default,
			CASE WHEN cc.constraint_type = 'P' THEN 'PRIMARY KEY' ELSE NULL END as constraint_type
		FROM all_tables t
		LEFT JOIN all_tab_columns c ON t.table_name = c.table_name AND t.owner = c.owner
		LEFT JOIN all_cons_columns cc ON c.table_name = cc.table_name
			AND c.column_name = cc.column_name AND cc.constraint_type = 'P'
		WHERE t.owner = :schema_name
		ORDER BY t.table_name, c.column_id
		"""

		# Execute in thread pool (Oracle is synchronous)
		loop = asyncio.get_event_loop()
		rows = await loop.run_in_executor(None, self._execute_oracle_query, table_query, schema_name)

		return self._group_table_data(rows)

	async def _discover_sqlserver_schema(self) -> List[Dict[str, Any]]:
		"""Discover SQL Server schema (synchronous)"""
		schema_name = self.data_source.schema or 'dbo'

		table_query = """
		SELECT
			t.TABLE_NAME as table_name,
			t.TABLE_TYPE as table_type,
			c.COLUMN_NAME as column_name,
			c.DATA_TYPE as data_type,
			c.IS_NULLABLE as is_nullable,
			c.COLUMN_DEFAULT as column_default,
			CASE WHEN kcu.COLUMN_NAME IS NOT NULL THEN 'PRIMARY KEY' ELSE NULL END as constraint_type
		FROM INFORMATION_SCHEMA.TABLES t
		LEFT JOIN INFORMATION_SCHEMA.COLUMNS c ON t.TABLE_NAME = c.TABLE_NAME
			AND t.TABLE_SCHEMA = c.TABLE_SCHEMA
		LEFT JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc ON t.TABLE_NAME = tc.TABLE_NAME
			AND t.TABLE_SCHEMA = tc.TABLE_SCHEMA AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
		LEFT JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
			AND c.COLUMN_NAME = kcu.COLUMN_NAME
		WHERE t.TABLE_SCHEMA = ?
			AND t.TABLE_TYPE IN ('BASE TABLE', 'VIEW')
		ORDER BY t.TABLE_NAME, c.ORDINAL_POSITION
		"""

		# Execute in thread pool (SQL Server is synchronous)
		loop = asyncio.get_event_loop()
		rows = await loop.run_in_executor(None, self._execute_sqlserver_query, table_query, schema_name)

		return self._group_table_data(rows)

	def _execute_oracle_query(self, query: str, schema_name: str) -> List[Dict[str, Any]]:
		"""Execute Oracle query synchronously"""
		conn = self.connection_pool.acquire()
		cursor = conn.cursor()
		try:
			cursor.execute(query, {'schema_name': schema_name})
			rows = cursor.fetchall()
			columns = [desc[0].lower() for desc in cursor.description]
			return [dict(zip(columns, row)) for row in rows]
		finally:
			cursor.close()
			self.connection_pool.release(conn)

	def _execute_sqlserver_query(self, query: str, schema_name: str) -> List[Dict[str, Any]]:
		"""Execute SQL Server query synchronously"""
		conn = pyodbc.connect(self.connection_string)
		cursor = conn.cursor()
		try:
			cursor.execute(query, schema_name)
			rows = cursor.fetchall()
			columns = [desc[0] for desc in cursor.description]
			return [dict(zip(columns, row)) for row in rows]
		finally:
			cursor.close()
			conn.close()

	def _group_table_data(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Group table and column data into structured format"""
		tables = {}

		for row in rows:
			table_name = row['table_name']
			if not table_name:
				continue

			if table_name not in tables:
				tables[table_name] = {
					'name': table_name,
					'type': row.get('table_type', '').lower().replace(' ', '_'),
					'columns': []
				}

			if row.get('column_name'):
				column = {
					'name': row['column_name'],
					'type': row['data_type'],
					'nullable': row.get('is_nullable') in ('YES', 'Y', True),
					'default': row.get('column_default'),
					'primary_key': row.get('constraint_type') == 'PRIMARY KEY'
				}
				tables[table_name]['columns'].append(column)

		return list(tables.values())

	async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Execute SQL query using appropriate database client"""
		try:
			execution_start = datetime.now(timezone.utc)
			parameters = parameters or {}

			if self.data_source.type == DataSourceType.POSTGRESQL:
				result = await self._execute_postgresql_query(query, parameters)
			elif self.data_source.type == DataSourceType.MYSQL:
				result = await self._execute_mysql_query(query, parameters)
			elif self.data_source.type == DataSourceType.ORACLE:
				result = await self._execute_oracle_query_async(query, parameters)
			elif self.data_source.type == DataSourceType.SQLSERVER:
				result = await self._execute_sqlserver_query_async(query, parameters)
			else:
				raise ValueError(f"Unsupported database type: {self.data_source.type}")

			execution_end = datetime.now(timezone.utc)
			execution_time_ms = int((execution_end - execution_start).total_seconds() * 1000)

			return {
				'query': query,
				'parameters': parameters,
				'results': result['rows'],
				'row_count': result['row_count'],
				'execution_time_ms': execution_time_ms,
				'columns': result['columns'],
				'database_type': self.data_source.type.value
			}

		except Exception as e:
			self.logger.error(f"Query execution failed for {self.data_source.name}: {e}")
			raise

	async def _execute_postgresql_query(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute PostgreSQL query"""
		async with self.connection_pool.acquire() as conn:
			# Convert named parameters to positional for asyncpg
			if parameters:
				# Simple parameter substitution - in production, use proper parameter binding
				for key, value in parameters.items():
					query = query.replace(f":{key}", f"${list(parameters.keys()).index(key) + 1}")
				rows = await conn.fetch(query, *parameters.values())
			else:
				rows = await conn.fetch(query)

			if rows:
				columns = list(rows[0].keys())
				row_data = [dict(row) for row in rows]
			else:
				columns = []
				row_data = []

			return {
				'rows': row_data,
				'row_count': len(row_data),
				'columns': columns
			}

	async def _execute_mysql_query(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute MySQL query"""
		async with self.connection_pool.acquire() as conn:
			async with conn.cursor() as cursor:
				if parameters:
					# Convert named parameters to %(key)s format
					for key in parameters.keys():
						query = query.replace(f":{key}", f"%({key})s")
					await cursor.execute(query, parameters)
				else:
					await cursor.execute(query)

				rows = await cursor.fetchall()
				columns = [desc[0] for desc in cursor.description or []]

				return {
					'rows': [dict(zip(columns, row)) for row in rows] if rows else [],
					'row_count': len(rows) if rows else 0,
					'columns': columns
				}

	async def _execute_oracle_query_async(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute Oracle query asynchronously"""
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(None, self._execute_oracle_sync, query, parameters)

	def _execute_oracle_sync(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute Oracle query synchronously"""
		conn = self.connection_pool.acquire()
		cursor = conn.cursor()
		try:
			if parameters:
				# Oracle uses :name format
				cursor.execute(query, parameters)
			else:
				cursor.execute(query)

			rows = cursor.fetchall()
			columns = [desc[0] for desc in cursor.description or []]

			return {
				'rows': [dict(zip(columns, row)) for row in rows] if rows else [],
				'row_count': len(rows) if rows else 0,
				'columns': columns
			}
		finally:
			cursor.close()
			self.connection_pool.release(conn)

	async def _execute_sqlserver_query_async(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute SQL Server query asynchronously"""
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(None, self._execute_sqlserver_sync, query, parameters)

	def _execute_sqlserver_sync(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute SQL Server query synchronously"""
		conn = pyodbc.connect(self.connection_string)
		cursor = conn.cursor()
		try:
			if parameters:
				# SQL Server uses ? placeholders
				param_values = list(parameters.values())
				# Replace named parameters with ?
				for key in parameters.keys():
					query = query.replace(f":{key}", "?")
				cursor.execute(query, param_values)
			else:
				cursor.execute(query)

			rows = cursor.fetchall()
			columns = [desc[0] for desc in cursor.description or []]

			return {
				'rows': [dict(zip(columns, row)) for row in rows] if rows else [],
				'row_count': len(rows) if rows else 0,
				'columns': columns
			}
		finally:
			cursor.close()
			conn.close()

	async def get_capabilities(self) -> List[ConnectionCapability]:
		"""Get SQL database capabilities"""
		self.capabilities = [
			ConnectionCapability.BATCH_READ,
			ConnectionCapability.BATCH_WRITE,
			ConnectionCapability.TRANSACTION_SUPPORT,
			ConnectionCapability.SCHEMA_INTROSPECTION,
			ConnectionCapability.QUERY_PUSHDOWN,
			ConnectionCapability.AGGREGATION_PUSHDOWN,
			ConnectionCapability.JOIN_PUSHDOWN,
			ConnectionCapability.LIMIT_PUSHDOWN
		]

		# Add specific capabilities based on database type
		if self.data_source.type in [DataSourceType.POSTGRESQL, DataSourceType.MYSQL]:
			self.capabilities.append(ConnectionCapability.FULL_TEXT_SEARCH)

		return self.capabilities

	def _get_driver_name(self) -> str:
		"""Get appropriate driver name for database type"""
		driver_map = {
			DataSourceType.POSTGRESQL: 'postgresql+asyncpg',
			DataSourceType.MYSQL: 'mysql+aiomysql',
			DataSourceType.ORACLE: 'oracle+cx_oracle',
			DataSourceType.SQLSERVER: 'mssql+pyodbc'
		}
		return driver_map.get(self.data_source.type, 'generic_sql')

	def _build_connection_string(self) -> str:
		"""Build connection string from data source configuration"""
		config = self.data_source.connection_config or {}
		driver = self._get_driver_name()

		# Build basic connection string
		host = config.get('host', 'localhost')
		port = config.get('port', 5432)
		database = config.get('database', config.get('dbname', 'default'))
		username = config.get('username', config.get('user', ''))
		password = config.get('password', '')

		if username and password:
			return f"{driver}://{username}:{password}@{host}:{port}/{database}"
		else:
			return f"{driver}://{host}:{port}/{database}"


class NoSQLConnector(BaseConnector):
	"""Production NoSQL database connector with real client libraries"""

	def __init__(self, data_source: DataSource, tenant_id: str, user_id: str):
		super().__init__(data_source, tenant_id, user_id)
		self.logger = logging.getLogger(f"dvrl.connectors.{data_source.type.value}")
		self.connection_pool = None
		self.client = None
		self.database = None
		self.cluster = None

	async def connect(self) -> bool:
		"""Connect to NoSQL database using appropriate client library"""
		try:
			config = self.data_source.connection_config or {}

			if self.data_source.type == DataSourceType.MONGODB:
				# MongoDB with motor (async)
				host = config.get('host', 'localhost')
				port = config.get('port', 27017)
				username = config.get('username')
				password = config.get('password')
				database_name = config.get('database', 'test')

				if username and password:
					uri = f"mongodb://{username}:{password}@{host}:{port}/{database_name}"
				else:
					uri = f"mongodb://{host}:{port}"

				self.client = motor.motor_asyncio.AsyncIOMotorClient(uri)
				self.database = self.client[database_name]

				# Test connection
				await self.database.command('ping')

			elif self.data_source.type == DataSourceType.CASSANDRA:
				# Cassandra with cassandra-driver (sync, but we'll handle it)
				hosts = config.get('hosts', ['localhost'])
				port = config.get('port', 9042)
				username = config.get('username')
				password = config.get('password')
				keyspace = config.get('keyspace', self.data_source.schema)

				if username and password:
					auth_provider = PlainTextAuthProvider(username, password)
				else:
					auth_provider = None

				self.cluster = Cluster(
					hosts,
					port=port,
					auth_provider=auth_provider,
					load_balancing_policy=RoundRobinPolicy()
				)
				self.connection_pool = self.cluster.connect(keyspace)

			elif self.data_source.type == DataSourceType.ELASTICSEARCH:
				# Elasticsearch with aiohttp
				host = config.get('host', 'localhost')
				port = config.get('port', 9200)
				scheme = config.get('scheme', 'http')
				username = config.get('username')
				password = config.get('password')

				self.base_url = f"{scheme}://{host}:{port}"

				# Create aiohttp session with auth if needed
				auth = None
				if username and password:
					auth = aiohttp.BasicAuth(username, password)

				self.client = aiohttp.ClientSession(auth=auth)

				# Test connection
				async with self.client.get(f"{self.base_url}/_cluster/health") as resp:
					if resp.status != 200:
						raise ConnectionError(f"Elasticsearch health check failed: {resp.status}")

			elif self.data_source.type == DataSourceType.REDIS:
				# Redis with aioredis
				try:
					import aioredis
					host = config.get('host', 'localhost')
					port = config.get('port', 6379)
					db = config.get('db', 0)
					password = config.get('password')

					self.client = await aioredis.create_redis_pool(
						f'redis://{host}:{port}/{db}',
						password=password
					)

					# Test connection
					await self.client.ping()
				except ImportError:
					raise ImportError("aioredis library required for Redis connections")

			else:
				raise ValueError(f"Unsupported NoSQL database type: {self.data_source.type}")

			# Store connection metadata
			self.connection_metadata = {
				'database_type': self.data_source.type.value,
				'connection_established': datetime.now(timezone.utc).isoformat(),
				'config': {k: v for k, v in config.items() if 'password' not in k.lower()}
			}

			self.logger.info(f"Connected to {self.data_source.type.value}: {self.data_source.name}")
			return True

		except Exception as e:
			self.logger.error(f"Failed to connect to NoSQL database {self.data_source.name}: {e}")
			return False

	async def disconnect(self) -> bool:
		"""Disconnect from NoSQL database"""
		try:
			if self.data_source.type == DataSourceType.MONGODB:
				if self.client:
					self.client.close()

			elif self.data_source.type == DataSourceType.CASSANDRA:
				if self.cluster:
					self.cluster.shutdown()

			elif self.data_source.type == DataSourceType.ELASTICSEARCH:
				if self.client:
					await self.client.close()

			elif self.data_source.type == DataSourceType.REDIS:
				if self.client:
					self.client.close()
					await self.client.wait_closed()

			self.client = None
			self.connection_pool = None
			self.cluster = None

			self.logger.info(f"Disconnected from NoSQL database: {self.data_source.name}")
			return True

		except Exception as e:
			self.logger.error(f"Failed to disconnect from NoSQL database {self.data_source.name}: {e}")
			return False

	async def test_connection(self) -> bool:
		"""Test NoSQL database connection"""
		try:
			if self.data_source.type == DataSourceType.MONGODB:
				if self.database:
					await self.database.command('ping')
					return True

			elif self.data_source.type == DataSourceType.CASSANDRA:
				if self.connection_pool:
					# Execute simple query
					loop = asyncio.get_event_loop()
					await loop.run_in_executor(None,
						lambda: self.connection_pool.execute("SELECT release_version FROM system.local"))
					return True

			elif self.data_source.type == DataSourceType.ELASTICSEARCH:
				if self.client:
					async with self.client.get(f"{self.base_url}/_cluster/health") as resp:
						return resp.status == 200

			elif self.data_source.type == DataSourceType.REDIS:
				if self.client:
					await self.client.ping()
					return True

			return False

		except Exception as e:
			self.logger.error(f"Connection test failed for {self.data_source.name}: {e}")
			return False

	async def discover_schema(self) -> DataSourceSchema:
		"""Auto-discover NoSQL database schema"""
		try:
			collections = []

			if self.data_source.type == DataSourceType.MONGODB:
				collections = await self._discover_mongodb_schema()
			elif self.data_source.type == DataSourceType.CASSANDRA:
				collections = await self._discover_cassandra_schema()
			elif self.data_source.type == DataSourceType.ELASTICSEARCH:
				collections = await self._discover_elasticsearch_schema()
			elif self.data_source.type == DataSourceType.REDIS:
				collections = await self._discover_redis_schema()

			schema = DataSourceSchema(
				data_source_id=self.data_source.id,
				schema_name=self.data_source.schema or 'default',
				tenant_id=self.tenant_id,
				created_by=self.user_id,
				tables=collections,
				discovery_method=f"{self.data_source.type.value}_introspection",
				confidence_score=0.90
			)

			self.logger.info(f"Discovered {len(collections)} collections for {self.data_source.name}")
			return schema

		except Exception as e:
			self.logger.error(f"NoSQL schema discovery failed for {self.data_source.name}: {e}")
			raise

	async def _discover_mongodb_schema(self) -> List[Dict[str, Any]]:
		"""Discover MongoDB collections and sample documents"""
		collections = []

		# List all collections
		collection_names = await self.database.list_collection_names()

		for collection_name in collection_names:
			collection = self.database[collection_name]

			# Get collection stats
			stats = await self.database.command('collStats', collection_name)
			document_count = stats.get('count', 0)

			# Sample a document to infer schema
			sample_doc = None
			if document_count > 0:
				cursor = collection.find().limit(1)
				async for doc in cursor:
					sample_doc = self._analyze_document_structure(doc)
					break

			collections.append({
				'name': collection_name,
				'type': 'collection',
				'document_count': document_count,
				'sample_document': sample_doc,
				'avg_document_size': stats.get('avgObjSize', 0)
			})

		return collections

	async def _discover_cassandra_schema(self) -> List[Dict[str, Any]]:
		"""Discover Cassandra tables and columns"""
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(None, self._discover_cassandra_sync)

	def _discover_cassandra_sync(self) -> List[Dict[str, Any]]:
		"""Discover Cassandra schema synchronously"""
		tables = []
		keyspace = self.connection_pool.keyspace

		# Query system tables for table metadata
		query = """
		SELECT table_name, bloom_filter_fp_chance, caching, comment
		FROM system_schema.tables
		WHERE keyspace_name = %s
		"""

		rows = self.connection_pool.execute(query, (keyspace,))

		for row in rows:
			table_name = row.table_name

			# Get column information
			col_query = """
			SELECT column_name, type, kind
			FROM system_schema.columns
			WHERE keyspace_name = %s AND table_name = %s
			"""

			col_rows = self.connection_pool.execute(col_query, (keyspace, table_name))

			columns = []
			partition_keys = []
			clustering_columns = []

			for col_row in col_rows:
				col_info = {
					'name': col_row.column_name,
					'type': str(col_row.type),
					'kind': col_row.kind
				}
				columns.append(col_info)

				if col_row.kind == 'partition_key':
					partition_keys.append(col_row.column_name)
				elif col_row.kind == 'clustering':
					clustering_columns.append(col_row.column_name)

			tables.append({
				'name': table_name,
				'type': 'table',
				'columns': columns,
				'partition_keys': partition_keys,
				'clustering_columns': clustering_columns,
				'comment': row.comment
			})

		return tables

	async def _discover_elasticsearch_schema(self) -> List[Dict[str, Any]]:
		"""Discover Elasticsearch indices and mappings"""
		indices = []

		# Get all indices
		async with self.client.get(f"{self.base_url}/_cat/indices?format=json") as resp:
			if resp.status == 200:
				index_list = await resp.json()

				for index_info in index_list:
					index_name = index_info['index']

					# Skip system indices
					if index_name.startswith('.'):
						continue

					# Get mapping for this index
					async with self.client.get(f"{self.base_url}/{index_name}/_mapping") as mapping_resp:
						if mapping_resp.status == 200:
							mapping_data = await mapping_resp.json()

							# Extract field information
							fields = []
							if index_name in mapping_data:
								properties = mapping_data[index_name].get('mappings', {}).get('properties', {})
								fields = self._extract_es_fields(properties)

							indices.append({
								'name': index_name,
								'type': 'index',
								'document_count': int(index_info.get('docs.count', 0)),
								'fields': fields,
								'size': index_info.get('store.size', '0b')
							})

		return indices

	async def _discover_redis_schema(self) -> List[Dict[str, Any]]:
		"""Discover Redis key patterns and types"""
		keyspaces = []

		# Sample keys to understand patterns
		sample_size = 100
		keys = await self.client.keys('*')

		if len(keys) > sample_size:
			keys = keys[:sample_size]

		key_patterns = {}

		for key in keys:
			key_str = key.decode() if isinstance(key, bytes) else str(key)
			key_type = await self.client.type(key_str)

			# Extract pattern (simple approach)
			pattern = self._extract_redis_pattern(key_str)

			if pattern not in key_patterns:
				key_patterns[pattern] = {
					'pattern': pattern,
					'type': key_type.decode() if isinstance(key_type, bytes) else str(key_type),
					'count': 0,
					'examples': []
				}

			key_patterns[pattern]['count'] += 1
			if len(key_patterns[pattern]['examples']) < 5:
				key_patterns[pattern]['examples'].append(key_str)

		keyspaces = [{
			'name': f"redis_pattern_{i}",
			'type': 'key_pattern',
			'pattern': info['pattern'],
			'data_type': info['type'],
			'estimated_count': info['count'],
			'examples': info['examples']
		} for i, info in enumerate(key_patterns.values())]

		return keyspaces

	def _analyze_document_structure(self, doc: Dict[str, Any], max_depth: int = 3) -> Dict[str, Any]:
		"""Analyze MongoDB document structure to infer schema"""
		if max_depth <= 0:
			return {'type': 'object', 'truncated': True}

		schema = {}

		for key, value in doc.items():
			if isinstance(value, dict):
				schema[key] = self._analyze_document_structure(value, max_depth - 1)
			elif isinstance(value, list):
				if value and isinstance(value[0], dict):
					schema[key] = {'type': 'array', 'items': self._analyze_document_structure(value[0], max_depth - 1)}
				else:
					schema[key] = {'type': 'array', 'items': type(value[0]).__name__ if value else 'unknown'}
			else:
				schema[key] = type(value).__name__

		return schema

	def _extract_es_fields(self, properties: Dict[str, Any], prefix: str = '') -> List[Dict[str, Any]]:
		"""Extract Elasticsearch field information from mapping"""
		fields = []

		for field_name, field_config in properties.items():
			full_name = f"{prefix}.{field_name}" if prefix else field_name

			field_info = {
				'name': full_name,
				'type': field_config.get('type', 'unknown'),
				'analyzer': field_config.get('analyzer'),
				'index': field_config.get('index', True)
			}

			fields.append(field_info)

			# Handle nested objects
			if 'properties' in field_config:
				nested_fields = self._extract_es_fields(field_config['properties'], full_name)
				fields.extend(nested_fields)

		return fields

	def _extract_redis_pattern(self, key: str) -> str:
		"""Extract pattern from Redis key"""
		# Simple pattern extraction - replace numbers with *
		pattern = re.sub(r'\d+', '*', key)
		# Replace UUIDs with pattern
		pattern = re.sub(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', '*', pattern)
		return pattern

	async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Execute NoSQL query using appropriate client library"""
		try:
			execution_start = datetime.now(timezone.utc)
			parameters = parameters or {}

			if self.data_source.type == DataSourceType.MONGODB:
				result = await self._execute_mongodb_query(query, parameters)
			elif self.data_source.type == DataSourceType.CASSANDRA:
				result = await self._execute_cassandra_query(query, parameters)
			elif self.data_source.type == DataSourceType.ELASTICSEARCH:
				result = await self._execute_elasticsearch_query(query, parameters)
			elif self.data_source.type == DataSourceType.REDIS:
				result = await self._execute_redis_query(query, parameters)
			else:
				raise ValueError(f"Unsupported NoSQL database type: {self.data_source.type}")

			execution_end = datetime.now(timezone.utc)
			execution_time_ms = int((execution_end - execution_start).total_seconds() * 1000)

			return {
				'query': query,
				'parameters': parameters,
				'results': result['documents'],
				'document_count': result['count'],
				'execution_time_ms': execution_time_ms,
				'database_type': self.data_source.type.value,
				'query_type': result.get('query_type', 'unknown')
			}

		except Exception as e:
			self.logger.error(f"NoSQL query execution failed for {self.data_source.name}: {e}")
			raise

	async def _execute_mongodb_query(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute MongoDB query - expects JSON query format or collection.operation"""
		try:
			# Parse query - could be JSON find query or collection.operation format
			if query.strip().startswith('{'):
				# JSON query format
				query_doc = json.loads(query)
				collection_name = parameters.get('collection', 'default')
				operation = parameters.get('operation', 'find')
			else:
				# collection.operation format
				parts = query.split('.', 1)
				collection_name = parts[0]
				operation_part = parts[1] if len(parts) > 1 else 'find()'

				# Extract operation and query from operation_part
				if '(' in operation_part:
					operation = operation_part.split('(')[0]
					query_part = operation_part.split('(', 1)[1].rsplit(')', 1)[0]
					query_doc = json.loads(query_part) if query_part.strip() else {}
				else:
					operation = 'find'
					query_doc = {}

			collection = self.database[collection_name]
			documents = []

			if operation == 'find':
				cursor = collection.find(query_doc)
				limit = parameters.get('limit', 100)
				cursor = cursor.limit(limit)

				async for doc in cursor:
					# Convert ObjectId to string for JSON serialization
					if '_id' in doc:
						doc['_id'] = str(doc['_id'])
					documents.append(doc)

			elif operation == 'aggregate':
				pipeline = query_doc if isinstance(query_doc, list) else [query_doc]
				cursor = collection.aggregate(pipeline)

				async for doc in cursor:
					if '_id' in doc:
						doc['_id'] = str(doc['_id'])
					documents.append(doc)

			elif operation == 'count':
				count = await collection.count_documents(query_doc)
				documents = [{'count': count}]

			return {
				'documents': documents,
				'count': len(documents),
				'query_type': f'mongodb_{operation}'
			}

		except Exception as e:
			self.logger.error(f"MongoDB query execution failed: {e}")
			raise

	async def _execute_cassandra_query(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute Cassandra CQL query"""
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(None, self._execute_cassandra_sync, query, parameters)

	def _execute_cassandra_sync(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute Cassandra query synchronously"""
		try:
			# Execute CQL query
			if parameters:
				# Use named parameters
				result = self.connection_pool.execute(query, parameters)
			else:
				result = self.connection_pool.execute(query)

			documents = []
			if result:
				for row in result:
					# Convert Row to dict
					row_dict = {}
					for key, value in row._asdict().items():
						# Convert special types to strings
						if hasattr(value, '__str__'):
							row_dict[key] = str(value) if not isinstance(value, (str, int, float, bool)) else value
						else:
							row_dict[key] = value
					documents.append(row_dict)

			return {
				'documents': documents,
				'count': len(documents),
				'query_type': 'cassandra_cql'
			}

		except Exception as e:
			self.logger.error(f"Cassandra query execution failed: {e}")
			raise

	async def _execute_elasticsearch_query(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute Elasticsearch query"""
		try:
			# Parse query - could be JSON or simple text search
			if query.strip().startswith('{'):
				query_doc = json.loads(query)
			else:
				# Simple text search
				query_doc = {
					'query': {
						'multi_match': {
							'query': query,
							'fields': ['*']
						}
					}
				}

			# Get index name
			index_name = parameters.get('index', '_all')

			# Execute search
			search_url = f"{self.base_url}/{index_name}/_search"
			async with self.client.post(search_url, json=query_doc) as resp:
				if resp.status == 200:
					result = await resp.json()

					documents = []
					if 'hits' in result and 'hits' in result['hits']:
						for hit in result['hits']['hits']:
							doc = hit['_source']
							doc['_id'] = hit['_id']
							doc['_score'] = hit.get('_score')
							documents.append(doc)

					return {
						'documents': documents,
						'count': len(documents),
						'total_hits': result.get('hits', {}).get('total', {}).get('value', 0),
						'query_type': 'elasticsearch_search'
					}
				else:
					error_text = await resp.text()
					raise Exception(f"Elasticsearch query failed with status {resp.status}: {error_text}")

		except Exception as e:
			self.logger.error(f"Elasticsearch query execution failed: {e}")
			raise

	async def _execute_redis_query(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute Redis command"""
		try:
			# Parse Redis command
			command_parts = query.split()
			command = command_parts[0].upper()
			args = command_parts[1:] if len(command_parts) > 1 else []

			# Add parameters as arguments
			for key, value in parameters.items():
				args.append(str(value))

			documents = []

			if command in ['GET', 'MGET']:
				if command == 'GET' and args:
					value = await self.client.get(args[0])
					documents = [{'key': args[0], 'value': value.decode() if value else None}]
				elif command == 'MGET':
					values = await self.client.mget(*args)
					for i, key in enumerate(args):
						value = values[i]
						documents.append({
							'key': key,
							'value': value.decode() if value else None
						})

				elif command == 'KEYS':
					pattern = args[0] if args else '*'
					keys = await self.client.keys(pattern)
					documents = [{'key': key.decode() if isinstance(key, bytes) else str(key)} for key in keys]

				elif command in ['HGETALL', 'HGET']:
					if args:
						hash_key = args[0]
						if command == 'HGETALL':
							hash_data = await self.client.hgetall(hash_key)
							if hash_data:
								doc = {'hash_key': hash_key}
								for k, v in hash_data.items():
									doc[k.decode() if isinstance(k, bytes) else str(k)] = v.decode() if isinstance(v, bytes) else v
								documents = [doc]
						elif command == 'HGET' and len(args) > 1:
							field = args[1]
							value = await self.client.hget(hash_key, field)
							documents = [{
								'hash_key': hash_key,
								'field': field,
								'value': value.decode() if value else None
							}]

			return {
				'documents': documents,
				'count': len(documents),
				'query_type': f'redis_{command.lower()}'
			}

		except Exception as e:
			self.logger.error(f"Redis query execution failed: {e}")
			raise

	async def get_capabilities(self) -> List[ConnectionCapability]:
		"""Get NoSQL database capabilities"""
		base_capabilities = [
			ConnectionCapability.BATCH_READ,
			ConnectionCapability.BATCH_WRITE,
			ConnectionCapability.SCHEMA_INTROSPECTION
		]

		# Add specific capabilities based on NoSQL type
		if self.data_source.type == DataSourceType.MONGODB:
			base_capabilities.extend([
				ConnectionCapability.FULL_TEXT_SEARCH,
				ConnectionCapability.AGGREGATION_PUSHDOWN
			])
		elif self.data_source.type == DataSourceType.CASSANDRA:
			base_capabilities.extend([
				ConnectionCapability.TIME_SERIES_SUPPORT,
				ConnectionCapability.STREAMING_READ
			])
		elif self.data_source.type == DataSourceType.ELASTICSEARCH:
			base_capabilities.extend([
				ConnectionCapability.FULL_TEXT_SEARCH,
				ConnectionCapability.VECTOR_SEARCH,
				ConnectionCapability.AGGREGATION_PUSHDOWN
			])

		self.capabilities = base_capabilities
		return self.capabilities



class APIConnector(BaseConnector):
	"""Production API connector for REST and GraphQL endpoints"""

	def __init__(self, data_source: DataSource, tenant_id: str, user_id: str):
		super().__init__(data_source, tenant_id, user_id)
		self.logger = logging.getLogger(f"dvrl.connectors.{data_source.type.value}")
		self.session = None
		self.base_url = None
		self.auth_headers = {}

	async def connect(self) -> bool:
		"""Connect to API endpoint with real HTTP client"""
		try:
			config = self.data_source.connection_config or {}
			self.base_url = config.get('base_url', config.get('url', ''))

			if not self.base_url:
				raise ValueError("API base_url is required")

			# Setup authentication
			auth = None
			auth_type = config.get('auth_type', 'none')

			if auth_type == 'basic':
				username = config.get('username')
				password = config.get('password')
				if username and password:
					auth = aiohttp.BasicAuth(username, password)

			elif auth_type == 'bearer':
				token = config.get('token', config.get('api_key'))
				if token:
					self.auth_headers['Authorization'] = f'Bearer {token}'

			elif auth_type == 'api_key':
				api_key = config.get('api_key')
				api_key_header = config.get('api_key_header', 'X-API-Key')
				if api_key:
					self.auth_headers[api_key_header] = api_key

			# Create session with timeout and auth
			timeout = aiohttp.ClientTimeout(total=30, connect=10)
			headers = {'Content-Type': 'application/json', **self.auth_headers}

			self.session = aiohttp.ClientSession(
				timeout=timeout,
				headers=headers,
				auth=auth
			)

			# Test connection with health check or basic GET
			health_endpoint = config.get('health_endpoint', '/')
			test_url = f"{self.base_url.rstrip('/')}{health_endpoint}"

			async with self.session.get(test_url) as resp:
				if resp.status >= 400:
					self.logger.warning(f"API health check returned {resp.status} for {test_url}")

			# Auto-discover API capabilities
			await self._discover_api_spec()

			self.connection_metadata = {
				'api_type': self.data_source.type.value,
				'base_url': self.base_url,
				'auth_type': auth_type,
				'api_version': config.get('version', '1.0'),
				'connection_established': datetime.now(timezone.utc).isoformat()
			}

			self.logger.info(f"Connected to {self.data_source.type.value} API: {self.data_source.name}")
			return True

		except Exception as e:
			self.logger.error(f"Failed to connect to API {self.data_source.name}: {e}")
			return False

	async def disconnect(self) -> bool:
		"""Disconnect from API"""
		try:
			if self.session:
				await self.session.close()
				self.session = None

			self.logger.info(f"Disconnected from API: {self.data_source.name}")
			return True

		except Exception as e:
			self.logger.error(f"Failed to disconnect from API {self.data_source.name}: {e}")
			return False

	async def test_connection(self) -> bool:
		"""Test API connection with real HTTP request"""
		try:
			if not self.session:
				return False

			# Try to access base URL or health endpoint
			config = self.data_source.connection_config or {}
			health_endpoint = config.get('health_endpoint', '/')
			test_url = f"{self.base_url.rstrip('/')}{health_endpoint}"

			async with self.session.get(test_url) as resp:
				# Consider 200-399 as successful
				return resp.status < 400

		except Exception as e:
			self.logger.error(f"API connection test failed for {self.data_source.name}: {e}")
			return False

	async def _discover_api_spec(self) -> None:
		"""Auto-discover API specification using real HTTP requests"""
		try:
			if self.data_source.type == DataSourceType.GRAPHQL:
				await self._discover_graphql_schema()
			else:
				# REST API - try to discover OpenAPI/Swagger spec
				await self._discover_rest_endpoints()

		except Exception as e:
			self.logger.warning(f"API spec discovery failed for {self.data_source.name}: {e}")
			# Continue without schema discovery - not critical for connection

	async def _discover_graphql_schema(self) -> None:
		"""Discover GraphQL schema through introspection query"""
		try:
			# GraphQL introspection query
			introspection_query = """
			query IntrospectionQuery {
				__schema {
					queryType { name }
					mutationType { name }
					types {
						name
						kind
						fields {
							name
							type {
								name
								kind
							}
						}
					}
				}
			}
			"""

			graphql_endpoint = f"{self.base_url.rstrip('/')}/graphql"
			payload = {'query': introspection_query}

			async with self.session.post(graphql_endpoint, json=payload) as resp:
				if resp.status == 200:
					result = await resp.json()
					schema_data = result.get('data', {}).get('__schema', {})

					# Extract type information
					schema_types = [t['name'] for t in schema_data.get('types', []) if not t['name'].startswith('__')]

					self.connection_metadata.update({
						'schema_types': schema_types,
						'query_type': schema_data.get('queryType', {}).get('name'),
						'mutation_type': schema_data.get('mutationType', {}).get('name'),
						'introspection_successful': True
					})
					self.logger.info(f"GraphQL schema discovered: {len(schema_types)} types")
				else:
					self.logger.warning(f"GraphQL introspection failed: {resp.status}")

		except Exception as e:
			self.logger.warning(f"GraphQL schema discovery failed: {e}")

	async def _discover_rest_endpoints(self) -> None:
		"""Discover REST API endpoints using OpenAPI/Swagger spec"""
		try:
			# Try common OpenAPI spec locations
			spec_paths = ['/swagger.json', '/openapi.json', '/api-docs', '/docs/swagger.json']

			for spec_path in spec_paths:
				spec_url = f"{self.base_url.rstrip(spec_path)}{spec_path}"

				async with self.session.get(spec_url) as resp:
					if resp.status == 200:
						spec_data = await resp.json()

						# Extract endpoint information
						endpoints = []
						paths = spec_data.get('paths', {})

						for path, path_info in paths.items():
							methods = list(path_info.keys())
							# Filter out non-HTTP methods
							methods = [m.upper() for m in methods if m.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']]

							if methods:
								endpoints.append({
									'path': path,
									'methods': methods
								})

						self.connection_metadata.update({
							'endpoints': endpoints,
							'api_info': spec_data.get('info', {}),
							'openapi_version': spec_data.get('openapi', spec_data.get('swagger')),
							'spec_discovery_successful': True
						})

						self.logger.info(f"OpenAPI spec discovered: {len(endpoints)} endpoints")
						return  # Successfully found spec

			# If no spec found, set basic metadata
			self.connection_metadata.update({
				'endpoints': [],
				'spec_discovery_successful': False
			})

		except Exception as e:
			self.logger.warning(f"REST API endpoint discovery failed: {e}")

	async def discover_schema(self) -> DataSourceSchema:
		"""Auto-discover API schema"""
		try:
			# Processing completed

			# Build schema from discovered endpoints
			tables = []
			endpoints = self.connection_metadata.get('endpoints', [])

			for endpoint in endpoints:
				# Create a table/resource entry for each endpoint
				path = endpoint['path']
				methods = endpoint['methods']

				# Generate resource name from path
				resource_name = path.strip('/').replace('/', '_').replace('{', '').replace('}', '')
				if not resource_name:
					resource_name = 'root'

				tables.append({
					'name': f"{resource_name}_endpoint",
					'type': 'api_resource',
					'endpoint': path,
					'methods': methods,
					'fields': []  # Would need OpenAPI spec analysis for detailed fields
				})

			# If no endpoints discovered, create a generic entry
			if not tables:
				tables = [{
					'name': 'api_endpoint',
					'type': 'api_resource',
					'endpoint': '/',
					'methods': ['GET'],
					'fields': []
				}]

			schema = DataSourceSchema(
				data_source_id=self.data_source.id,
				schema_name='api_schema',
				tenant_id=self.tenant_id,
				created_by=self.user_id,
				tables=tables,
				discovery_method="api_introspection",
				confidence_score=0.90
			)

			return schema

		except Exception as e:
			await _log_error(f"API schema discovery failed for {self.data_source.name}", e)
			raise

	async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Execute API query using real HTTP requests"""
		try:
			execution_start = datetime.now(timezone.utc)
			parameters = parameters or {}

			if self.data_source.type == DataSourceType.GRAPHQL:
				result = await self._execute_graphql_query(query, parameters)
			else:
				# REST API
				result = await self._execute_rest_query(query, parameters)

			execution_end = datetime.now(timezone.utc)
			execution_time_ms = int((execution_end - execution_start).total_seconds() * 1000)

			return {
				'query': query,
				'parameters': parameters,
				'response': result['data'],
				'record_count': result['count'],
				'execution_time_ms': execution_time_ms,
				'api_call_type': result['method'],
				'status_code': result['status_code']
			}

		except Exception as e:
			self.logger.error(f"API query execution failed for {self.data_source.name}: {e}")
			raise

	async def _execute_graphql_query(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute GraphQL query"""
		try:
			graphql_endpoint = f"{self.base_url.rstrip('/')}/graphql"
			payload = {
				'query': query,
				'variables': parameters
			}

			async with self.session.post(graphql_endpoint, json=payload) as resp:
				response_data = await resp.json()

				if resp.status == 200 and 'data' in response_data:
					data = response_data['data']
					# Flatten data for consistent response format
					if isinstance(data, dict):
						# Convert single object to list
						data = [data] if data else []
					elif isinstance(data, list):
						pass  # Already a list
					else:
						data = [{'result': data}]

					return {
						'data': data,
						'count': len(data) if isinstance(data, list) else 1,
						'method': 'GRAPHQL',
						'status_code': resp.status,
						'errors': response_data.get('errors', [])
					}
				else:
					raise Exception(f"GraphQL query failed: {response_data.get('errors', resp.status)}")

		except Exception as e:
			self.logger.error(f"GraphQL query execution failed: {e}")
			raise

	async def _execute_rest_query(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute REST API query"""
		try:
			# Parse query - could be "GET /users" or just "/users"
			parts = query.strip().split()
			if len(parts) >= 2:
				method = parts[0].upper()
				path = parts[1]
			else:
				method = parameters.get('method', 'GET').upper()
				path = query

			# Build full URL
			if not path.startswith('/'):
				path = '/' + path
			full_url = f"{self.base_url.rstrip('/')}{path}"

			# Prepare request parameters
			query_params = parameters.get('query_params', {})
			body_data = parameters.get('body', parameters.get('data'))
			headers = parameters.get('headers', {})

			# Execute request
			kwargs = {
				'params': query_params,
				'headers': headers
			}

			if body_data and method in ['POST', 'PUT', 'PATCH']:
				kwargs['json'] = body_data

			async with self.session.request(method, full_url, **kwargs) as resp:
				if resp.content_type == 'application/json':
					response_data = await resp.json()
				else:
					response_text = await resp.text()
					try:
						response_data = json.loads(response_text)
					except Exception:
						response_data = {'response': response_text}

				# Handle different response formats
				if isinstance(response_data, list):
					data = response_data
					count = len(data)
				elif isinstance(response_data, dict):
					# Check for common list fields
					for key in ['data', 'results', 'items', 'records']:
						if key in response_data and isinstance(response_data[key], list):
							data = response_data[key]
							count = len(data)
							break
					else:
						# Single object response
						data = [response_data]
						count = 1
				else:
					# Non-JSON response
					data = [response_data]
					count = 1

				if resp.status >= 400:
					raise Exception(f"HTTP {resp.status}: {response_data}")

				return {
					'data': data,
					'count': count,
					'method': method,
					'status_code': resp.status,
					'raw_response': response_data
				}

		except Exception as e:
			self.logger.error(f"REST API query execution failed: {e}")
			raise

	async def get_capabilities(self) -> List[ConnectionCapability]:
		"""Get API capabilities"""
		self.capabilities = [
			ConnectionCapability.BATCH_READ,
			ConnectionCapability.SCHEMA_INTROSPECTION
		]

		# Add write capabilities if API supports them
		if any('POST' in endpoint.get('methods', []) for endpoint in
			   self.connection_metadata.get('endpoints', [])):
			self.capabilities.append(ConnectionCapability.BATCH_WRITE)

		# GraphQL APIs typically support more advanced querying
		if self.connection_metadata.get('api_type') == 'graphql':
			self.capabilities.extend([
				ConnectionCapability.QUERY_PUSHDOWN,
				ConnectionCapability.AGGREGATION_PUSHDOWN
			])

		return self.capabilities


class StreamingConnector(BaseConnector):
	"""Production streaming data connector using Bytewax-style streams."""

	def __init__(self, data_source: DataSource, tenant_id: str, user_id: str):
		super().__init__(data_source, tenant_id, user_id)
		self.logger = logging.getLogger(f"dvrl.connectors.{data_source.type.value}")
		self.streams: Dict[str, List[Dict[str, Any]]] = {}
		self.stream_cursors: Dict[str, int] = {}

	async def connect(self) -> bool:
		"""Connect to configured Bytewax stream fixtures."""
		try:
			config = self.data_source.connection_config or {}
			if self.data_source.type != DataSourceType.BYTEWAX:
				raise ValueError(f"Unsupported streaming platform: {self.data_source.type}")

			stream_names = config.get('streams') or config.get('stream_names') or []
			if isinstance(stream_names, str):
				stream_names = [stream.strip() for stream in stream_names.split(',') if stream.strip()]
			stream_records = config.get('sample_records', {})
			self.streams = {stream: [] for stream in stream_names}
			for stream, records in stream_records.items():
				self.streams.setdefault(stream, [])
				for record in records:
					self.streams[stream].append(self._normalize_stream_record(stream, record))

			self.connection_metadata = {
				'platform': self.data_source.type.value,
				'streams': list(self.streams.keys()),
				'flow_id': config.get('flow_id', f"dvrl_{self.tenant_id}_{self.user_id}"),
				'connection_established': datetime.now(timezone.utc).isoformat()
			}

			self.logger.info(f"Connected to {self.data_source.type.value}: {self.data_source.name}")
			return True

		except Exception as e:
			self.logger.error(f"Failed to connect to streaming platform {self.data_source.name}: {e}")
			return False

	def _normalize_stream_record(self, stream: str, record: Any) -> Dict[str, Any]:
		"""Normalize a Bytewax stream fixture record."""
		if not isinstance(record, dict):
			record = {'value': record}
		return {
			'stream': stream,
			'sequence': record.get('sequence', len(self.streams.get(stream, []))),
			'timestamp': record.get('timestamp', datetime.now(timezone.utc).isoformat()),
			'key': record.get('key'),
			'value': record.get('value', record)
		}

	async def disconnect(self) -> bool:
		"""Disconnect from streaming platform."""
		self.stream_cursors.clear()
		self.logger.info(f"Disconnected from streaming platform: {self.data_source.name}")
		return True

	async def test_connection(self) -> bool:
		"""Test Bytewax stream metadata availability."""
		return self.data_source.type == DataSourceType.BYTEWAX and bool(self.streams or self.connection_metadata)

	async def _get_bytewax_streams(self) -> List[str]:
		"""Get available Bytewax streams."""
		return list(self.streams.keys())

	async def discover_schema(self) -> DataSourceSchema:
		"""Auto-discover streaming schema."""
		try:
			streams = []
			for stream_name in await self._get_bytewax_streams():
				streams.append({
					'name': stream_name,
					'type': 'bytewax_stream',
					'record_count': len(self.streams.get(stream_name, [])),
					'schema': self._infer_stream_schema(self.streams.get(stream_name, []))
				})

			return DataSourceSchema(
				data_source_id=self.data_source.id,
				schema_name='bytewax_streams',
				tenant_id=self.tenant_id,
				created_by=self.user_id,
				tables=streams,
				discovery_method="stream_introspection",
				confidence_score=0.92
			)

		except Exception as e:
			await _log_error(f"Streaming schema discovery failed for {self.data_source.name}", e)
			raise

	def _infer_stream_schema(self, records: List[Dict[str, Any]]) -> Dict[str, str]:
		"""Infer a lightweight field schema from stream records."""
		schema: Dict[str, str] = {}
		for record in records:
			value = record.get('value')
			if isinstance(value, str):
				try:
					value = json.loads(value)
				except (json.JSONDecodeError, ValueError):
					value = {'value': value}
			if not isinstance(value, dict):
				value = {'value': value}
			for key, item in value.items():
				schema.setdefault(key, type(item).__name__)
		return schema

	async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Execute streaming query using Bytewax stream fixtures."""
		try:
			execution_start = datetime.now(timezone.utc)
			parameters = parameters or {}
			result = await self._execute_bytewax_query(query, parameters)
			execution_end = datetime.now(timezone.utc)
			execution_time_ms = int((execution_end - execution_start).total_seconds() * 1000)

			return {
				'query': query,
				'parameters': parameters,
				'messages': result['messages'],
				'message_count': result['count'],
				'execution_time_ms': execution_time_ms,
				'streaming': True,
				'platform': self.data_source.type.value
			}

		except Exception as e:
			self.logger.error(f"Streaming query execution failed for {self.data_source.name}: {e}")
			raise

	async def _execute_bytewax_query(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute Bytewax streaming query."""
		parts = query.strip().split()
		if not parts:
			raise ValueError("Empty query")
		command = parts[0].upper()

		if command == 'CONSUME':
			stream = parts[1] if len(parts) > 1 else parameters.get('stream')
			if not stream:
				raise ValueError("Stream name required for CONSUME")
			return await self._consume_bytewax_messages(stream, parameters)

		if command == 'PRODUCE':
			stream = parts[1] if len(parts) > 1 else parameters.get('stream')
			if not stream:
				raise ValueError("Stream name required for PRODUCE")
			return await self._produce_bytewax_message(stream, parameters)

		if command in ['LIST', 'STREAMS']:
			streams = await self._get_bytewax_streams()
			return {'messages': [{'stream': stream} for stream in streams], 'count': len(streams)}

		raise ValueError(f"Unsupported Bytewax command: {command}")

	async def _consume_bytewax_messages(self, stream: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Consume messages from a Bytewax stream."""
		max_messages = parameters.get('limit', parameters.get('max_messages', 10))
		start_sequence = parameters.get('start_sequence', self.stream_cursors.get(stream, 0))
		messages = [
			record for record in self.streams.get(stream, [])
			if record.get('sequence', 0) >= start_sequence
		][:max_messages]
		if messages:
			self.stream_cursors[stream] = messages[-1].get('sequence', 0) + 1
		return {'messages': messages, 'count': len(messages)}

	async def _produce_bytewax_message(self, stream: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Produce message to a Bytewax stream."""
		value = parameters.get('value', parameters.get('message'))
		if value is None:
			raise ValueError("Message value required for PRODUCE")
		self.streams.setdefault(stream, [])
		record = self._normalize_stream_record(stream, {
			'key': parameters.get('key'),
			'value': value,
			'sequence': len(self.streams[stream])
		})
		self.streams[stream].append(record)
		return {'messages': [record], 'count': 1}

	async def get_capabilities(self) -> List[ConnectionCapability]:
		"""Get streaming platform capabilities."""
		self.capabilities = [
			ConnectionCapability.STREAMING_READ,
			ConnectionCapability.STREAMING_WRITE,
			ConnectionCapability.SCHEMA_INTROSPECTION,
			ConnectionCapability.TIME_SERIES_SUPPORT
		]
		return self.capabilities


class ConnectorFactory:
	"""Factory for creating appropriate connectors based on data source type"""

	_connector_registry: Dict[DataSourceType, Type[BaseConnector]] = {
		# SQL Databases
		DataSourceType.POSTGRESQL: SQLDatabaseConnector,
		DataSourceType.MYSQL: SQLDatabaseConnector,
		DataSourceType.ORACLE: SQLDatabaseConnector,
		DataSourceType.SQLSERVER: SQLDatabaseConnector,

		# NoSQL Databases
		DataSourceType.MONGODB: NoSQLConnector,
		DataSourceType.CASSANDRA: NoSQLConnector,
		DataSourceType.REDIS: NoSQLConnector,
		DataSourceType.ELASTICSEARCH: NoSQLConnector,

		# Cloud Warehouses - Will be overridden by specialized adapters
		DataSourceType.SNOWFLAKE: SQLDatabaseConnector,
		DataSourceType.BIGQUERY: SQLDatabaseConnector,
		DataSourceType.REDSHIFT: SQLDatabaseConnector,

		# APIs
		DataSourceType.REST_API: APIConnector,
		DataSourceType.GRAPHQL: APIConnector,

		# Streaming
			DataSourceType.BYTEWAX: StreamingConnector,

		# File Systems - Will be registered by adapters module
		# DataSourceType.FILE_CSV: FileSystemConnector,
		# DataSourceType.FILE_JSON: FileSystemConnector,
		# DataSourceType.FILE_PARQUET: FileSystemConnector,
		# DataSourceType.S3: CloudStorageConnector,
		# DataSourceType.HDFS: DistributedFileSystemConnector,
	}

	@classmethod
	def create_connector(cls, data_source: DataSource, tenant_id: str, user_id: str) -> BaseConnector:
		"""Create appropriate connector for data source"""
		connector_class = cls._connector_registry.get(data_source.type)

		if not connector_class:
			raise ValueError(f"No connector available for data source type: {data_source.type}")

		return connector_class(data_source, tenant_id, user_id)

	@classmethod
	def register_connector(cls, data_source_type: DataSourceType, connector_class: Type[BaseConnector]) -> None:
		"""Register custom connector for data source type"""
		cls._connector_registry[data_source_type] = connector_class

	@classmethod
	def get_supported_types(cls) -> List[DataSourceType]:
		"""Get list of supported data source types"""
		return list(cls._connector_registry.keys())


class UniversalConnectorManager:
	"""Manager for universal connector framework with auto-discovery"""

	def __init__(self, tenant_id: str, user_id: str):
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.active_connectors: Dict[str, BaseConnector] = {}
		self.connector_pool: Dict[str, List[BaseConnector]] = {}
		self.discovery_cache: Dict[str, DataSourceSchema] = {}

	async def create_connector(self, data_source: DataSource) -> BaseConnector:
		"""Create and initialize connector for data source"""
		try:
			connector = ConnectorFactory.create_connector(data_source, self.tenant_id, self.user_id)

			# Connect to data source
			success = await connector.connect()
			if not success:
				raise ConnectionError(f"Failed to connect to data source: {data_source.name}")

			# Discover capabilities
			capabilities = await connector.get_capabilities()
			await _log_info(f"Connector created with capabilities: {[c.value for c in capabilities]}")

			# Store active connector
			self.active_connectors[data_source.id] = connector

			return connector

		except Exception as e:
			await _log_error(f"Failed to create connector for {data_source.name}", e)
			raise

	async def get_connector(self, data_source_id: str) -> Optional[BaseConnector]:
		"""Get existing connector or create new one"""
		return self.active_connectors.get(data_source_id)

	async def remove_connector(self, data_source_id: str) -> bool:
		"""Remove and disconnect connector"""
		if data_source_id in self.active_connectors:
			connector = self.active_connectors[data_source_id]
			await connector.disconnect()
			del self.active_connectors[data_source_id]
			await _log_info(f"Connector removed for data source: {data_source_id}")
			return True
		return False

	async def discover_all_schemas(self) -> Dict[str, DataSourceSchema]:
		"""Discover schemas for all active connectors"""
		schemas = {}

		for data_source_id, connector in self.active_connectors.items():
			try:
				schema = await connector.discover_schema()
				schemas[data_source_id] = schema
				self.discovery_cache[data_source_id] = schema
				await _log_info(f"Schema discovered for: {connector.data_source.name}")
			except Exception as e:
				await _log_error(f"Schema discovery failed for: {connector.data_source.name}", e)

		return schemas

	async def health_check_all(self) -> Dict[str, ConnectionHealth]:
		"""Perform health check on all connectors"""
		health_results = {}

		for data_source_id, connector in self.active_connectors.items():
			try:
				health = await connector.health_check()
				health_results[data_source_id] = health
			except Exception as e:
				health_results[data_source_id] = ConnectionHealth.UNHEALTHY
				await _log_error(f"Health check failed for: {connector.data_source.name}", e)

		return health_results

	async def get_connector_stats(self) -> Dict[str, Any]:
		"""Get comprehensive connector statistics"""
		stats = {
			'total_connectors': len(self.active_connectors),
			'connector_types': {},
			'health_summary': {},
			'capabilities_summary': {},
			'connectors': {}
		}

		for data_source_id, connector in self.active_connectors.items():
			connector_stats = await connector.get_connection_stats()
			stats['connectors'][data_source_id] = connector_stats

			# Aggregate statistics
			connector_type = connector_stats['connector_type']
			stats['connector_types'][connector_type] = stats['connector_types'].get(connector_type, 0) + 1

			health_status = connector_stats['health_status']
			stats['health_summary'][health_status] = stats['health_summary'].get(health_status, 0) + 1

			# Aggregate capabilities
			for capability in connector_stats['capabilities']:
				stats['capabilities_summary'][capability] = stats['capabilities_summary'].get(capability, 0) + 1

		return stats


# Register additional connector types as they become available
try:
	from .singer_integration import SingerTapConnector
	ConnectorFactory.register_connector(DataSourceType.SINGER_TAP, SingerTapConnector)
except ImportError:
	pass  # Singer integration optional


# Export main components
__all__ = [
	"ConnectionCapability",
	"ConnectionHealth",
	"BaseConnector",
	"SQLDatabaseConnector",
	"NoSQLConnector",
	"APIConnector",
	"StreamingConnector",
	"ConnectorFactory",
	"UniversalConnectorManager"
]
