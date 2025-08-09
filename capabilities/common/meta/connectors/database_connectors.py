#!/usr/bin/env python3
"""
APG Metadata Management - Database Connectors
Connectors for discovering metadata from various database systems

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import asyncpg
import aiomysql
import pymongo
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from .base_connector import (
	BaseConnector, ConnectorConfig, DiscoveryResult, AssetMetadata, 
	ColumnMetadata, ConnectorType, DataType, should_include_asset
)


class PostgreSQLConnector(BaseConnector):
	"""PostgreSQL metadata discovery connector"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.DATABASE
		self.source_system = "postgresql"
		self.connection: Optional[asyncpg.Connection] = None
	
	async def connect(self) -> bool:
		"""Connect to PostgreSQL database"""
		try:
			self.connection = await asyncpg.connect(
				host=self.config.host,
				port=self.config.port or 5432,
				user=self.config.username,
				password=self.config.password,
				database=self.config.database,
				timeout=self.config.connection_timeout
			)
			self.is_connected = True
			return True
		except Exception as e:
			await self._log_error(f"Failed to connect to PostgreSQL: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Disconnect from PostgreSQL"""
		if self.connection:
			await self.connection.close()
			self.connection = None
			self.is_connected = False
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test PostgreSQL connection"""
		if not await self.connect():
			return {"status": "error", "message": "Failed to connect"}
		
		try:
			version = await self.connection.fetchval("SELECT version()")
			await self.disconnect()
			return {
				"status": "success",
				"database_type": "postgresql",
				"version": version,
				"message": "Connection successful"
			}
		except Exception as e:
			await self.disconnect()
			return {"status": "error", "message": str(e)}
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover all tables and views in PostgreSQL database"""
		result = DiscoveryResult(
			connector_type=self.connector_type,
			source_system=self.source_system
		)
		
		if not await self.connect():
			result.add_error("Failed to connect to database")
			result.complete_discovery()
			return result
		
		try:
			# Get all tables and views
			query = """
			SELECT 
				schemaname as schema_name,
				tablename as table_name,
				'table' as asset_type
			FROM pg_tables 
			WHERE schemaname NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
			
			UNION ALL
			
			SELECT 
				schemaname as schema_name,
				viewname as table_name,
				'view' as asset_type
			FROM pg_views
			WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
			
			ORDER BY schema_name, table_name
			"""
			
			rows = await self.connection.fetch(query)
			
			for row in rows:
				schema_name = row['schema_name']
				table_name = row['table_name']
				asset_type = row['asset_type']
				full_name = f"{schema_name}.{table_name}"
				
				# Check include/exclude patterns
				if not should_include_asset(table_name, self.config.include_patterns, self.config.exclude_patterns):
					continue
				
				try:
					# Get detailed schema information
					asset_metadata = await self.get_asset_schema(full_name)
					if asset_metadata:
						result.add_asset(asset_metadata)
				except Exception as e:
					result.add_error(f"Failed to get schema for {full_name}: {str(e)}")
			
		except Exception as e:
			result.add_error(f"Discovery failed: {str(e)}")
		finally:
			await self.disconnect()
		
		result.complete_discovery()
		return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get detailed schema information for PostgreSQL table/view"""
		try:
			parts = asset_name.split('.')
			if len(parts) != 2:
				return None
			
			schema_name, table_name = parts
			
			# Get table information
			table_info_query = """
			SELECT 
				obj_description(c.oid) as table_comment,
				CASE WHEN c.relkind = 'v' THEN 'view' ELSE 'table' END as asset_type
			FROM pg_class c
			JOIN pg_namespace n ON n.oid = c.relnamespace
			WHERE n.nspname = $1 AND c.relname = $2
			"""
			
			table_info = await self.connection.fetchrow(table_info_query, schema_name, table_name)
			if not table_info:
				return None
			
			# Get column information
			columns_query = """
			SELECT 
				a.attname as column_name,
				format_type(a.atttypid, a.atttypmod) as data_type,
				a.attnotnull as not_null,
				a.atthasdef as has_default,
				pg_get_expr(d.adbin, d.adrelid) as default_value,
				col_description(a.attrelid, a.attnum) as column_comment,
				CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary_key,
				CASE WHEN fk.column_name IS NOT NULL THEN true ELSE false END as is_foreign_key,
				fk.foreign_table_name,
				fk.foreign_column_name
			FROM pg_attribute a
			LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
			LEFT JOIN (
				SELECT ku.column_name
				FROM information_schema.table_constraints tc
				JOIN information_schema.key_column_usage ku ON tc.constraint_name = ku.constraint_name
				WHERE tc.constraint_type = 'PRIMARY KEY' 
				AND tc.table_schema = $1 AND tc.table_name = $2
			) pk ON pk.column_name = a.attname
			LEFT JOIN (
				SELECT 
					kcu.column_name,
					ccu.table_name AS foreign_table_name,
					ccu.column_name AS foreign_column_name
				FROM information_schema.table_constraints tc
				JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
				JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name
				WHERE tc.constraint_type = 'FOREIGN KEY' 
				AND tc.table_schema = $1 AND tc.table_name = $2
			) fk ON fk.column_name = a.attname
			WHERE a.attrelid = (
				SELECT c.oid 
				FROM pg_class c 
				JOIN pg_namespace n ON n.oid = c.relnamespace 
				WHERE n.nspname = $1 AND c.relname = $2
			)
			AND a.attnum > 0 
			AND NOT a.attisdropped
			ORDER BY a.attnum
			"""
			
			column_rows = await self.connection.fetch(columns_query, schema_name, table_name)
			
			columns = []
			for col_row in column_rows:
				data_type = self._map_postgres_type(col_row['data_type'])
				
				column = ColumnMetadata(
					name=col_row['column_name'],
					data_type=data_type,
					is_nullable=not col_row['not_null'],
					is_primary_key=col_row['is_primary_key'],
					is_foreign_key=col_row['is_foreign_key'],
					foreign_key_table=col_row.get('foreign_table_name'),
					foreign_key_column=col_row.get('foreign_column_name'),
					default_value=col_row.get('default_value'),
					description=col_row.get('column_comment')
				)
				
				# Add profiling if enabled
				if self.config.enable_profiling:
					try:
						sample_data = await self.sample_asset_data(asset_name, self.config.max_sample_rows)
						if sample_data:
							column_values = [row.get(col_row['column_name']) for row in sample_data]
							profiled_column = await self.profile_column(asset_name, col_row['column_name'], column_values)
							
							# Merge profiling data
							column.distinct_count = profiled_column.distinct_count
							column.null_count = profiled_column.null_count
							column.null_percentage = profiled_column.null_percentage
							column.min_value = profiled_column.min_value
							column.max_value = profiled_column.max_value
							column.avg_value = profiled_column.avg_value
							column.sample_values = profiled_column.sample_values
							column.classification_hints = profiled_column.classification_hints
							column.contains_pii = profiled_column.contains_pii
							column.contains_phi = profiled_column.contains_phi
					except:
						pass  # Continue without profiling if it fails
				
				columns.append(column)
			
			# Get row count for tables
			row_count = None
			if table_info['asset_type'] == 'table':
				try:
					count_query = f'SELECT COUNT(*) FROM "{schema_name}"."{table_name}"'
					row_count = await self.connection.fetchval(count_query)
				except:
					pass  # Row count not critical
			
			asset_metadata = AssetMetadata(
				name=table_name,
				asset_type=table_info['asset_type'],
				source_system=self.source_system,
				schema_name=schema_name,
				full_name=asset_name,
				description=table_info.get('table_comment'),
				columns=columns,
				column_count=len(columns),
				row_count=row_count,
				properties={
					"database": self.config.database,
					"schema": schema_name
				}
			)
			
			# Estimate quality score
			asset_metadata.estimated_quality_score = self._estimate_quality_score(asset_metadata)
			
			return asset_metadata
			
		except Exception as e:
			await self._log_error(f"Failed to get schema for {asset_name}: {str(e)}")
			return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Sample data from PostgreSQL table/view"""
		try:
			parts = asset_name.split('.')
			if len(parts) != 2:
				return []
			
			schema_name, table_name = parts
			query = f'SELECT * FROM "{schema_name}"."{table_name}" LIMIT $1'
			
			rows = await self.connection.fetch(query, limit)
			return [dict(row) for row in rows]
			
		except Exception as e:
			await self._log_error(f"Failed to sample data from {asset_name}: {str(e)}")
			return []
	
	def _map_postgres_type(self, postgres_type: str) -> DataType:
		"""Map PostgreSQL data types to standard DataType enum"""
		type_lower = postgres_type.lower()
		
		if any(t in type_lower for t in ['varchar', 'char', 'text', 'string']):
			return DataType.STRING
		elif any(t in type_lower for t in ['int', 'serial', 'bigserial']):
			return DataType.INTEGER
		elif any(t in type_lower for t in ['real', 'double', 'numeric', 'decimal', 'float']):
			return DataType.FLOAT
		elif 'boolean' in type_lower or 'bool' in type_lower:
			return DataType.BOOLEAN
		elif 'date' in type_lower and 'time' not in type_lower:
			return DataType.DATE
		elif any(t in type_lower for t in ['timestamp', 'datetime']):
			return DataType.DATETIME
		elif any(t in type_lower for t in ['json', 'jsonb']):
			return DataType.JSON
		elif any(t in type_lower for t in ['array', '[]']):
			return DataType.ARRAY
		elif any(t in type_lower for t in ['bytea', 'blob']):
			return DataType.BINARY
		else:
			return DataType.UNKNOWN


class MySQLConnector(BaseConnector):
	"""MySQL metadata discovery connector"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.DATABASE
		self.source_system = "mysql"
		self.connection = None
	
	async def connect(self) -> bool:
		"""Connect to MySQL database"""
		try:
			self.connection = await aiomysql.connect(
				host=self.config.host,
				port=self.config.port or 3306,
				user=self.config.username,
				password=self.config.password,
				db=self.config.database,
				connect_timeout=self.config.connection_timeout
			)
			self.is_connected = True
			return True
		except Exception as e:
			await self._log_error(f"Failed to connect to MySQL: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Disconnect from MySQL"""
		if self.connection:
			self.connection.close()
			self.connection = None
			self.is_connected = False
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test MySQL connection"""
		if not await self.connect():
			return {"status": "error", "message": "Failed to connect"}
		
		try:
			cursor = self.connection.cursor()
			await cursor.execute("SELECT VERSION()")
			version = await cursor.fetchone()
			await cursor.close()
			await self.disconnect()
			
			return {
				"status": "success",
				"database_type": "mysql",
				"version": version[0] if version else "unknown",
				"message": "Connection successful"
			}
		except Exception as e:
			await self.disconnect()
			return {"status": "error", "message": str(e)}
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover all tables and views in MySQL database"""
		result = DiscoveryResult(
			connector_type=self.connector_type,
			source_system=self.source_system
		)
		
		if not await self.connect():
			result.add_error("Failed to connect to database")
			result.complete_discovery()
			return result
		
		try:
			cursor = self.connection.cursor()
			
			# Get all tables and views
			query = """
			SELECT 
				TABLE_SCHEMA as schema_name,
				TABLE_NAME as table_name,
				TABLE_TYPE as table_type
			FROM information_schema.TABLES 
			WHERE TABLE_SCHEMA = %s
			ORDER BY TABLE_NAME
			"""
			
			await cursor.execute(query, (self.config.database,))
			rows = await cursor.fetchall()
			
			for row in rows:
				schema_name, table_name, table_type = row
				asset_type = 'view' if 'VIEW' in table_type else 'table'
				full_name = f"{schema_name}.{table_name}"
				
				# Check include/exclude patterns
				if not should_include_asset(table_name, self.config.include_patterns, self.config.exclude_patterns):
					continue
				
				try:
					# Get detailed schema information
					asset_metadata = await self.get_asset_schema(full_name)
					if asset_metadata:
						result.add_asset(asset_metadata)
				except Exception as e:
					result.add_error(f"Failed to get schema for {full_name}: {str(e)}")
			
			await cursor.close()
			
		except Exception as e:
			result.add_error(f"Discovery failed: {str(e)}")
		finally:
			await self.disconnect()
		
		result.complete_discovery()
		return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get detailed schema information for MySQL table/view"""
		try:
			parts = asset_name.split('.')
			if len(parts) != 2:
				return None
			
			schema_name, table_name = parts
			cursor = self.connection.cursor()
			
			# Get table information
			table_query = """
			SELECT 
				TABLE_COMMENT as table_comment,
				CASE WHEN TABLE_TYPE LIKE '%VIEW%' THEN 'view' ELSE 'table' END as asset_type,
				TABLE_ROWS as row_count
			FROM information_schema.TABLES
			WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
			"""
			
			await cursor.execute(table_query, (schema_name, table_name))
			table_info = await cursor.fetchone()
			
			if not table_info:
				await cursor.close()
				return None
			
			# Get column information
			columns_query = """
			SELECT 
				COLUMN_NAME as column_name,
				DATA_TYPE as data_type,
				IS_NULLABLE as is_nullable,
				COLUMN_KEY as column_key,
				COLUMN_DEFAULT as default_value,
				COLUMN_COMMENT as column_comment,
				CHARACTER_MAXIMUM_LENGTH as max_length,
				NUMERIC_PRECISION as precision,
				NUMERIC_SCALE as scale
			FROM information_schema.COLUMNS
			WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
			ORDER BY ORDINAL_POSITION
			"""
			
			await cursor.execute(columns_query, (schema_name, table_name))
			column_rows = await cursor.fetchall()
			
			columns = []
			for col_row in column_rows:
				(column_name, data_type, is_nullable, column_key, 
				 default_value, column_comment, max_length, precision, scale) = col_row
				
				mapped_type = self._map_mysql_type(data_type)
				
				column = ColumnMetadata(
					name=column_name,
					data_type=mapped_type,
					is_nullable=is_nullable == 'YES',
					is_primary_key=column_key == 'PRI',
					is_foreign_key=column_key == 'MUL',
					max_length=max_length,
					precision=precision,
					scale=scale,
					default_value=default_value,
					description=column_comment
				)
				
				columns.append(column)
			
			await cursor.close()
			
			table_comment, asset_type, row_count = table_info
			
			asset_metadata = AssetMetadata(
				name=table_name,
				asset_type=asset_type,
				source_system=self.source_system,
				schema_name=schema_name,
				full_name=asset_name,
				description=table_comment,
				columns=columns,
				column_count=len(columns),
				row_count=row_count,
				properties={
					"database": self.config.database,
					"schema": schema_name
				}
			)
			
			# Estimate quality score
			asset_metadata.estimated_quality_score = self._estimate_quality_score(asset_metadata)
			
			return asset_metadata
			
		except Exception as e:
			await self._log_error(f"Failed to get schema for {asset_name}: {str(e)}")
			return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Sample data from MySQL table/view"""
		try:
			parts = asset_name.split('.')
			if len(parts) != 2:
				return []
			
			schema_name, table_name = parts
			cursor = self.connection.cursor(aiomysql.DictCursor)
			
			query = f"SELECT * FROM `{schema_name}`.`{table_name}` LIMIT %s"
			await cursor.execute(query, (limit,))
			rows = await cursor.fetchall()
			await cursor.close()
			
			return rows
			
		except Exception as e:
			await self._log_error(f"Failed to sample data from {asset_name}: {str(e)}")
			return []
	
	def _map_mysql_type(self, mysql_type: str) -> DataType:
		"""Map MySQL data types to standard DataType enum"""
		type_lower = mysql_type.lower()
		
		if any(t in type_lower for t in ['varchar', 'char', 'text', 'longtext', 'mediumtext']):
			return DataType.STRING
		elif any(t in type_lower for t in ['int', 'bigint', 'smallint', 'tinyint']):
			return DataType.INTEGER
		elif any(t in type_lower for t in ['float', 'double', 'decimal', 'numeric']):
			return DataType.FLOAT
		elif any(t in type_lower for t in ['boolean', 'bool', 'bit']):
			return DataType.BOOLEAN
		elif 'date' in type_lower and 'time' not in type_lower:
			return DataType.DATE
		elif any(t in type_lower for t in ['datetime', 'timestamp']):
			return DataType.DATETIME
		elif 'json' in type_lower:
			return DataType.JSON
		elif any(t in type_lower for t in ['blob', 'binary', 'varbinary']):
			return DataType.BINARY
		else:
			return DataType.UNKNOWN


class MongoDBConnector(BaseConnector):
	"""MongoDB metadata discovery connector"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.DATABASE
		self.source_system = "mongodb"
		self.client = None
		self.database = None
	
	async def connect(self) -> bool:
		"""Connect to MongoDB"""
		try:
			connection_string = self.config.connection_string
			if not connection_string:
				connection_string = f"mongodb://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port or 27017}/{self.config.database}"
			
			self.client = pymongo.MongoClient(
				connection_string,
				serverSelectionTimeoutMS=self.config.connection_timeout * 1000
			)
			
			# Test connection
			self.client.admin.command('ismaster')
			
			self.database = self.client[self.config.database]
			self.is_connected = True
			return True
		except Exception as e:
			await self._log_error(f"Failed to connect to MongoDB: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Disconnect from MongoDB"""
		if self.client:
			self.client.close()
			self.client = None
			self.database = None
			self.is_connected = False
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test MongoDB connection"""
		if not await self.connect():
			return {"status": "error", "message": "Failed to connect"}
		
		try:
			server_info = self.client.server_info()
			await self.disconnect()
			return {
				"status": "success",
				"database_type": "mongodb",
				"version": server_info.get('version', 'unknown'),
				"message": "Connection successful"
			}
		except Exception as e:
			await self.disconnect()
			return {"status": "error", "message": str(e)}
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover all collections in MongoDB database"""
		result = DiscoveryResult(
			connector_type=self.connector_type,
			source_system=self.source_system
		)
		
		if not await self.connect():
			result.add_error("Failed to connect to database")
			result.complete_discovery()
			return result
		
		try:
			collection_names = self.database.list_collection_names()
			
			for collection_name in collection_names:
				# Check include/exclude patterns
				if not should_include_asset(collection_name, self.config.include_patterns, self.config.exclude_patterns):
					continue
				
				try:
					# Get detailed schema information
					asset_metadata = await self.get_asset_schema(collection_name)
					if asset_metadata:
						result.add_asset(asset_metadata)
				except Exception as e:
					result.add_error(f"Failed to get schema for {collection_name}: {str(e)}")
			
		except Exception as e:
			result.add_error(f"Discovery failed: {str(e)}")
		finally:
			await self.disconnect()
		
		result.complete_discovery()
		return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get schema information for MongoDB collection"""
		try:
			collection = self.database[asset_name]
			
			# Get collection statistics
			stats = collection.aggregate([
				{"$collStats": {"storageStats": {}}}
			])
			
			collection_stats = next(stats, {})
			
			# Sample documents to infer schema
			sample_docs = list(collection.find().limit(100))
			
			# Infer schema from sample documents
			columns = self._infer_mongo_schema(sample_docs)
			
			# Get document count
			doc_count = collection.estimated_document_count()
			
			asset_metadata = AssetMetadata(
				name=asset_name,
				asset_type="collection",
				source_system=self.source_system,
				schema_name=self.config.database,
				full_name=f"{self.config.database}.{asset_name}",
				columns=columns,
				column_count=len(columns),
				row_count=doc_count,
				size_bytes=collection_stats.get('size', 0),
				properties={
					"database": self.config.database,
					"is_capped": collection_stats.get('capped', False),
					"index_count": len(list(collection.list_indexes()))
				}
			)
			
			# Estimate quality score
			asset_metadata.estimated_quality_score = self._estimate_quality_score(asset_metadata)
			
			return asset_metadata
			
		except Exception as e:
			await self._log_error(f"Failed to get schema for {asset_name}: {str(e)}")
			return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Sample data from MongoDB collection"""
		try:
			collection = self.database[asset_name]
			documents = list(collection.find().limit(limit))
			
			# Convert ObjectId and other non-serializable types to strings
			for doc in documents:
				for key, value in doc.items():
					if hasattr(value, '__str__') and not isinstance(value, (str, int, float, bool, list, dict)):
						doc[key] = str(value)
			
			return documents
			
		except Exception as e:
			await self._log_error(f"Failed to sample data from {asset_name}: {str(e)}")
			return []
	
	def _infer_mongo_schema(self, sample_docs: List[Dict[str, Any]]) -> List[ColumnMetadata]:
		"""Infer schema from MongoDB document samples"""
		if not sample_docs:
			return []
		
		# Collect all field names and their types
		field_info = {}
		
		for doc in sample_docs:
			for field_name, value in doc.items():
				if field_name not in field_info:
					field_info[field_name] = {
						'types': set(),
						'null_count': 0,
						'total_count': 0,
						'sample_values': []
					}
				
				field_info[field_name]['total_count'] += 1
				
				if value is None:
					field_info[field_name]['null_count'] += 1
				else:
					field_info[field_name]['types'].add(type(value).__name__)
					if len(field_info[field_name]['sample_values']) < 10:
						field_info[field_name]['sample_values'].append(value)
		
		columns = []
		for field_name, info in field_info.items():
			# Determine data type from collected types
			data_type = self._infer_mongo_field_type(info['types'])
			
			null_percentage = (info['null_count'] / info['total_count']) * 100 if info['total_count'] > 0 else 0
			
			column = ColumnMetadata(
				name=field_name,
				data_type=data_type,
				is_nullable=info['null_count'] > 0,
				null_count=info['null_count'],
				null_percentage=round(null_percentage, 2),
				sample_values=[str(v) for v in info['sample_values'][:5]]
			)
			
			columns.append(column)
		
		return columns
	
	def _infer_mongo_field_type(self, python_types: Set[str]) -> DataType:
		"""Infer DataType from Python types found in MongoDB documents"""
		if not python_types:
			return DataType.UNKNOWN
		
		if 'str' in python_types:
			return DataType.STRING
		elif 'int' in python_types:
			return DataType.INTEGER
		elif 'float' in python_types:
			return DataType.FLOAT
		elif 'bool' in python_types:
			return DataType.BOOLEAN
		elif 'datetime' in python_types:
			return DataType.DATETIME
		elif 'list' in python_types:
			return DataType.ARRAY
		elif 'dict' in python_types:
			return DataType.OBJECT
		else:
			return DataType.UNKNOWN


# Additional database connectors can be implemented following the same pattern
# For now, we'll create placeholder connectors that can be extended

class SnowflakeConnector(BaseConnector):
	"""Placeholder for Snowflake connector - to be implemented"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.DATABASE
		self.source_system = "snowflake"
	
	async def connect(self) -> bool:
		"""Connect to Snowflake warehouse"""
		try:
			import snowflake.connector
			from snowflake.connector import DictCursor
			
			# Extract connection parameters
			connection_params = {
				'account': self.config.additional_params.get('account'),
				'user': self.config.username or self.config.additional_params.get('user'),
				'password': self.config.password,
				'warehouse': self.config.additional_params.get('warehouse'),
				'database': self.config.database or 'DEMO_DB',
				'schema': self.config.schema or 'PUBLIC',
				'role': self.config.additional_params.get('role', 'PUBLIC')
			}
			
			# Validate required parameters
			required_params = ['account', 'user', 'password', 'warehouse']
			for param in required_params:
				if not connection_params.get(param):
					return False
			
			# Create connection
			self.connection = snowflake.connector.connect(**connection_params)
			self.cursor = self.connection.cursor(DictCursor)
			
			# Test connection
			self.cursor.execute("SELECT CURRENT_VERSION()")
			self.cursor.fetchone()
			
			return True
			
		except ImportError:
			return False
		except Exception as e:
			await self._log_error(f"Snowflake connection failed: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Disconnect from Snowflake"""
		try:
			if hasattr(self, 'cursor') and self.cursor:
				self.cursor.close()
			if hasattr(self, 'connection') and self.connection:
				self.connection.close()
		except Exception as e:
			await self._log_error(f"Snowflake disconnect failed: {str(e)}")
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test Snowflake connection"""
		try:
			is_connected = await self.connect()
			if is_connected:
				await self.disconnect()
				return {"status": "success", "message": "Connected to Snowflake successfully"}
			else:
				return {"status": "error", "message": "Failed to connect to Snowflake"}
		except Exception as e:
			return {"status": "error", "message": f"Connection test failed: {str(e)}"}
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover assets from Snowflake warehouse"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		
		try:
			if not await self.connect():
				result.add_error("Failed to connect to Snowflake")
				result.complete_discovery()
				return result
			
			# Discover tables and views
			tables_query = """
			SELECT 
				table_catalog as database_name,
				table_schema as schema_name,
				table_name,
				table_type,
				created as created_at,
				last_altered as updated_at,
				row_count,
				bytes,
				comment as description
			FROM information_schema.tables
			WHERE table_schema NOT IN ('INFORMATION_SCHEMA')
			ORDER BY table_schema, table_name
			LIMIT 1000
			"""
			
			self.cursor.execute(tables_query)
			tables = self.cursor.fetchall()
			
			for table in tables:
				# Get column information
				columns_query = f"""
				SELECT 
					column_name,
					data_type,
					is_nullable,
					column_default,
					comment as description
				FROM information_schema.columns
				WHERE table_catalog = '{table['DATABASE_NAME']}'
				AND table_schema = '{table['SCHEMA_NAME']}'
				AND table_name = '{table['TABLE_NAME']}'
				ORDER BY ordinal_position
				"""
				
				self.cursor.execute(columns_query)
				columns = self.cursor.fetchall()
				
				# Create column metadata objects
				column_metadata = []
				for col in columns:
					column_metadata.append(ColumnMetadata(
						name=col['COLUMN_NAME'],
						data_type=self._map_snowflake_type(col['DATA_TYPE']),
						is_nullable=col['IS_NULLABLE'] == 'YES',
						default_value=col.get('COLUMN_DEFAULT'),
						description=col.get('DESCRIPTION')
					))
				
				metadata = AssetMetadata(
					name=table['TABLE_NAME'],
					description=table.get('DESCRIPTION') or f"Snowflake {table['TABLE_TYPE'].lower()}",
					asset_type=table['TABLE_TYPE'].lower(),
					source_system=self.source_system,
					schema_name=table['SCHEMA_NAME'],
					full_name=f"{table['SCHEMA_NAME']}.{table['TABLE_NAME']}",
					columns=column_metadata,
					column_count=len(columns),
					row_count=table.get('ROW_COUNT', 0),
					size_bytes=table.get('BYTES', 0),
					properties={
						'database': table['DATABASE_NAME'],
						'schema': table['SCHEMA_NAME'],
						'table_type': table['TABLE_TYPE'],
						'warehouse': self.config.additional_params.get('warehouse'),
						'created_at': table.get('CREATED_AT'),
						'updated_at': table.get('UPDATED_AT')
					}
				)
				
				result.add_asset(metadata)
			
			result.complete_discovery()
			await self.disconnect()
			
		except Exception as e:
			result.add_error(f"Snowflake discovery failed: {str(e)}")
			result.complete_discovery()
			await self.disconnect()
		
		return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get detailed schema for a specific Snowflake asset"""
		try:
			if not await self.connect():
				return None
			
			# Parse asset name (format: schema.table)
			if '.' in asset_name:
				schema_name, table_name = asset_name.split('.', 1)
			else:
				schema_name = self.config.schema or 'PUBLIC'
				table_name = asset_name
			
			# Get table information
			table_query = f"""
			SELECT 
				table_catalog as database_name,
				table_schema as schema_name,
				table_name,
				table_type,
				created as created_at,
				last_altered as updated_at,
				row_count,
				bytes,
				comment as description
			FROM information_schema.tables
			WHERE table_schema = '{schema_name}'
			AND table_name = '{table_name}'
			"""
			
			self.cursor.execute(table_query)
			table = self.cursor.fetchone()
			
			if not table:
				await self.disconnect()
				return None
			
			# Get column information
			columns_query = f"""
			SELECT 
				column_name,
				data_type,
				is_nullable,
				column_default,
				comment as description,
				ordinal_position
			FROM information_schema.columns
			WHERE table_schema = '{schema_name}'
			AND table_name = '{table_name}'
			ORDER BY ordinal_position
			"""
			
			self.cursor.execute(columns_query)
			columns = self.cursor.fetchall()
			
			# Create column metadata objects
			column_metadata = []
			for col in columns:
				column_metadata.append(ColumnMetadata(
					name=col['COLUMN_NAME'],
					data_type=self._map_snowflake_type(col['DATA_TYPE']),
					is_nullable=col['IS_NULLABLE'] == 'YES',
					default_value=col.get('COLUMN_DEFAULT'),
					description=col.get('DESCRIPTION')
				))
			
			metadata = AssetMetadata(
				name=table['TABLE_NAME'],
				description=table.get('DESCRIPTION') or f"Snowflake {table['TABLE_TYPE'].lower()}",
				asset_type=table['TABLE_TYPE'].lower(),
				source_system=self.source_system,
				schema_name=table['SCHEMA_NAME'],
				full_name=f"{table['SCHEMA_NAME']}.{table['TABLE_NAME']}",
				columns=column_metadata,
				column_count=len(columns),
				row_count=table.get('ROW_COUNT', 0),
				size_bytes=table.get('BYTES', 0),
				properties={
					'database': table['DATABASE_NAME'],
					'schema': table['SCHEMA_NAME'],
					'table_type': table['TABLE_TYPE'],
					'warehouse': self.config.additional_params.get('warehouse'),
					'created_at': table.get('CREATED_AT'),
					'updated_at': table.get('UPDATED_AT')
				}
			)
			
			await self.disconnect()
			return metadata
			
		except Exception as e:
			await self._log_error(f"Failed to get Snowflake asset schema: {str(e)}")
			await self.disconnect()
			return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Get sample data from Snowflake table"""
		try:
			if not await self.connect():
				return []
			
			# Parse asset name (format: schema.table)
			if '.' in asset_name:
				schema_name, table_name = asset_name.split('.', 1)
			else:
				schema_name = self.config.schema or 'PUBLIC'
				table_name = asset_name
			
			database_name = self.config.database or 'DEMO_DB'
			sample_query = f"""
			SELECT * FROM {database_name}.{schema_name}.{table_name}
			LIMIT {min(limit, 1000)}
			"""
			
			self.cursor.execute(sample_query)
			results = self.cursor.fetchall()
			
			await self.disconnect()
			return [dict(row) for row in results]
			
		except Exception as e:
			await self._log_error(f"Failed to sample Snowflake data: {str(e)}")
			await self.disconnect()
			return []

	def _map_snowflake_type(self, snowflake_type: str) -> DataType:
		"""Map Snowflake data types to standard DataType enum"""
		snowflake_type = snowflake_type.upper()
		
		# Numeric types
		if snowflake_type in ['NUMBER', 'DECIMAL', 'NUMERIC']:
			return DataType.FLOAT
		elif snowflake_type in ['INTEGER', 'INT', 'BIGINT', 'SMALLINT', 'TINYINT']:
			return DataType.INTEGER
		elif snowflake_type in ['FLOAT', 'FLOAT4', 'FLOAT8', 'DOUBLE', 'DOUBLE PRECISION', 'REAL']:
			return DataType.FLOAT
		
		# String types
		elif snowflake_type in ['VARCHAR', 'CHAR', 'CHARACTER', 'STRING', 'TEXT']:
			return DataType.STRING
		elif snowflake_type in ['BINARY', 'VARBINARY']:
			return DataType.BINARY
		
		# Date/Time types
		elif snowflake_type in ['DATE']:
			return DataType.DATE
		elif snowflake_type in ['TIME', 'TIMESTAMP', 'TIMESTAMP_LTZ', 'TIMESTAMP_NTZ', 'TIMESTAMP_TZ']:
			return DataType.DATETIME
		
		# Boolean type
		elif snowflake_type in ['BOOLEAN']:
			return DataType.BOOLEAN
		
		# Semi-structured types
		elif snowflake_type in ['VARIANT', 'OBJECT', 'ARRAY']:
			return DataType.JSON
		
		# Default to unknown for unrecognized types
		else:
			return DataType.UNKNOWN


# Additional Database Connectors - Placeholder Implementations

class OracleConnector(BaseConnector):
	"""Oracle database metadata discovery connector"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.DATABASE
		self.source_system = "oracle"
	
	async def connect(self) -> bool:
		"""Connect to Oracle database"""
		try:
			# Would use cx_Oracle or oracledb in real implementation
			await self._log_error("Oracle connector not fully implemented yet")
			return False
		except Exception as e:
			await self._log_error(f"Oracle connection failed: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Disconnect from Oracle"""
		self.is_connected = False
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test Oracle connection"""
		return {"status": "error", "message": "Oracle connector not implemented"}
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover Oracle database assets"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		result.add_error("Oracle connector not fully implemented")
		result.complete_discovery()
		return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get Oracle asset schema"""
		await self._log_error("Oracle connector not implemented")
		return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Sample Oracle asset data"""
		await self._log_error("Oracle connector not implemented")
		return []


class SQLServerConnector(BaseConnector):
	"""SQL Server database metadata discovery connector"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.DATABASE
		self.source_system = "sqlserver"
	
	async def connect(self) -> bool:
		"""Connect to SQL Server database"""
		try:
			# Would use pyodbc or aioodbc in real implementation
			await self._log_error("SQL Server connector not fully implemented yet")
			return False
		except Exception as e:
			await self._log_error(f"SQL Server connection failed: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Disconnect from SQL Server"""
		self.is_connected = False
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test SQL Server connection"""
		return {"status": "error", "message": "SQL Server connector not implemented"}
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover SQL Server database assets"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		result.add_error("SQL Server connector not fully implemented")
		result.complete_discovery()
		return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get SQL Server asset schema"""
		await self._log_error("SQL Server connector not implemented")
		return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Sample SQL Server asset data"""
		await self._log_error("SQL Server connector not implemented")
		return []


class RedisConnector(BaseConnector):
	"""Redis NoSQL database metadata discovery connector"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.DATABASE
		self.source_system = "redis"
	
	async def connect(self) -> bool:
		"""Connect to Redis database"""
		try:
			# Would use aioredis in real implementation
			await self._log_error("Redis connector not fully implemented yet")
			return False
		except Exception as e:
			await self._log_error(f"Redis connection failed: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Disconnect from Redis"""
		self.is_connected = False
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test Redis connection"""
		return {"status": "error", "message": "Redis connector not implemented"}
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover Redis database assets"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		result.add_error("Redis connector not fully implemented")
		result.complete_discovery()
		return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get Redis asset schema"""
		await self._log_error("Redis connector not implemented")
		return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Sample Redis asset data"""
		await self._log_error("Redis connector not implemented")
		return []


class BigQueryConnector(BaseConnector):
	"""Google BigQuery metadata discovery connector"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.DATABASE
		self.source_system = "bigquery"
	
	async def connect(self) -> bool:
		"""Connect to BigQuery"""
		try:
			# Would use google-cloud-bigquery in real implementation
			await self._log_error("BigQuery connector not fully implemented yet")
			return False
		except Exception as e:
			await self._log_error(f"BigQuery connection failed: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Disconnect from BigQuery"""
		self.is_connected = False
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test BigQuery connection"""
		return {"status": "error", "message": "BigQuery connector not implemented"}
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover BigQuery assets"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		result.add_error("BigQuery connector not fully implemented")
		result.complete_discovery()
		return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get BigQuery asset schema"""
		await self._log_error("BigQuery connector not implemented")
		return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Sample BigQuery asset data"""
		await self._log_error("BigQuery connector not implemented")
		return []