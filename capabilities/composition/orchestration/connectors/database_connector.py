"""
APG Workflow Orchestration Database Connectors

High-performance database connectors for PostgreSQL, MongoDB, and other
database systems with connection pooling, transaction management, and caching.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import logging
import json

import asyncpg
import motor.motor_asyncio
from pymongo import MongoClient
from pydantic import BaseModel, Field, ConfigDict, validator

from .base_connector import BaseConnector, ConnectorConfiguration

logger = logging.getLogger(__name__)

class DatabaseConfiguration(ConnectorConfiguration):
	"""Base configuration for database connectors."""
	
	host: str = Field(..., description="Database host")
	port: int = Field(..., ge=1, le=65535, description="Database port")
	database: str = Field(..., min_length=1, description="Database name")
	username: str = Field(..., min_length=1, description="Database username")
	password: str = Field(..., min_length=1, description="Database password")
	ssl_mode: str = Field(default="prefer", regex="^(disable|allow|prefer|require|verify-ca|verify-full)$")
	max_connections: int = Field(default=20, ge=1, le=100)
	min_connections: int = Field(default=2, ge=1, le=50)
	connection_timeout: int = Field(default=30, ge=1, le=300)
	command_timeout: int = Field(default=60, ge=1, le=3600)
	enable_logging: bool = Field(default=False)
	pool_recycle_time: int = Field(default=3600, ge=300)

class PostgreSQLConfiguration(DatabaseConfiguration):
	"""PostgreSQL-specific configuration."""
	
	port: int = Field(default=5432, ge=1, le=65535)
	application_name: str = Field(default="APG-WorkflowOrchestration")
	server_settings: Dict[str, str] = Field(default_factory=dict)
	prepared_statement_cache_size: int = Field(default=100, ge=0, le=1000)
	prepared_statement_name_func: Optional[str] = Field(default=None)

class MongoDBConfiguration(DatabaseConfiguration):
	"""MongoDB-specific configuration."""
	
	port: int = Field(default=27017, ge=1, le=65535)
	auth_source: str = Field(default="admin")
	replica_set: Optional[str] = Field(default=None)
	read_preference: str = Field(default="primary", regex="^(primary|primaryPreferred|secondary|secondaryPreferred|nearest)$")
	write_concern_w: Union[int, str] = Field(default=1)
	write_concern_j: bool = Field(default=True)
	read_concern_level: str = Field(default="local", regex="^(local|available|majority|linearizable|snapshot)$")
	max_pool_size: int = Field(default=50, ge=1, le=500)
	min_pool_size: int = Field(default=5, ge=1, le=50)

class DatabaseConnector(BaseConnector):
	"""Base database connector with common functionality."""
	
	def __init__(self, config: DatabaseConfiguration):
		super().__init__(config)
		self.config: DatabaseConfiguration = config
		self.connection_pool = None
		self.active_connections = 0
		self.transaction_count = 0
		self.query_cache: Dict[str, Any] = {}
	
	async def execute_query(
		self,
		query: str,
		parameters: Optional[Union[List, Dict, tuple]] = None,
		fetch: str = "all",
		timeout: Optional[int] = None
	) -> Any:
		"""Execute database query with caching and error handling."""
		
		operation_params = {
			"query": query,
			"parameters": parameters or [],
			"fetch": fetch,
			"timeout": timeout or self.config.command_timeout
		}
		
		return await self.execute_request("query", operation_params)
	
	async def execute_transaction(
		self,
		queries: List[Dict[str, Any]],
		timeout: Optional[int] = None
	) -> List[Any]:
		"""Execute multiple queries in a transaction."""
		
		operation_params = {
			"queries": queries,
			"timeout": timeout or self.config.command_timeout
		}
		
		result = await self.execute_request("transaction", operation_params)
		return result.get("results", [])
	
	def get_connection_stats(self) -> Dict[str, Any]:
		"""Get database connection statistics."""
		base_stats = self.get_metrics()
		base_stats.update({
			"active_connections": self.active_connections,
			"transaction_count": self.transaction_count,
			"max_connections": self.config.max_connections,
			"min_connections": self.config.min_connections
		})
		return base_stats

class PostgreSQLAdapter(DatabaseConnector):
	"""High-performance PostgreSQL connector."""
	
	def __init__(self, config: PostgreSQLConfiguration):
		super().__init__(config)
		self.config: PostgreSQLConfiguration = config
		self.connection_pool: Optional[asyncpg.Pool] = None
	
	async def _connect(self) -> None:
		"""Create PostgreSQL connection pool."""
		
		# Build connection string
		connection_params = {
			"host": self.config.host,
			"port": self.config.port,
			"database": self.config.database,
			"user": self.config.username,
			"password": self.config.password,
			"ssl": self.config.ssl_mode,
			"server_settings": {
				"application_name": self.config.application_name,
				**self.config.server_settings
			}
		}
		
		# Create connection pool
		self.connection_pool = await asyncpg.create_pool(
			min_size=self.config.min_connections,
			max_size=self.config.max_connections,
			command_timeout=self.config.command_timeout,
			**connection_params
		)
		
		logger.info(self._log_connector_info("PostgreSQL connection pool created"))
	
	async def _disconnect(self) -> None:
		"""Close PostgreSQL connection pool."""
		if self.connection_pool:
			await self.connection_pool.close()
			self.connection_pool = None
		
		logger.info(self._log_connector_info("PostgreSQL connection pool closed"))
	
	async def _execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute PostgreSQL operation."""
		
		if operation == "query":
			return await self._execute_query(parameters)
		elif operation == "transaction":
			return await self._execute_transaction(parameters)
		else:
			raise ValueError(f"Unsupported operation: {operation}")
	
	async def _execute_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute single PostgreSQL query."""
		
		query = params["query"]
		query_params = params.get("parameters", [])
		fetch = params.get("fetch", "all")
		timeout = params.get("timeout", self.config.command_timeout)
		
		async with self.connection_pool.acquire() as connection:
			self.active_connections += 1
			
			try:
				if fetch == "all":
					result = await connection.fetch(query, *query_params, timeout=timeout)
					return {
						"rows": [dict(row) for row in result],
						"row_count": len(result)
					}
				elif fetch == "one":
					result = await connection.fetchrow(query, *query_params, timeout=timeout)
					return {
						"row": dict(result) if result else None,
						"row_count": 1 if result else 0
					}
				elif fetch == "value":
					result = await connection.fetchval(query, *query_params, timeout=timeout)
					return {
						"value": result,
						"row_count": 1 if result is not None else 0
					}
				elif fetch == "execute":
					result = await connection.execute(query, *query_params, timeout=timeout)
					return {
						"status": result,
						"row_count": self._parse_execute_result(result)
					}
				else:
					raise ValueError(f"Unsupported fetch type: {fetch}")
			
			finally:
				self.active_connections -= 1
	
	async def _execute_transaction(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute PostgreSQL transaction."""
		
		queries = params["queries"]
		timeout = params.get("timeout", self.config.command_timeout)
		results = []
		
		async with self.connection_pool.acquire() as connection:
			self.active_connections += 1
			self.transaction_count += 1
			
			try:
				async with connection.transaction():
					for query_info in queries:
						query = query_info["query"]
						query_params = query_info.get("parameters", [])
						fetch = query_info.get("fetch", "execute")
						
						if fetch == "all":
							result = await connection.fetch(query, *query_params, timeout=timeout)
							results.append({"rows": [dict(row) for row in result], "row_count": len(result)})
						elif fetch == "one":
							result = await connection.fetchrow(query, *query_params, timeout=timeout)
							results.append({"row": dict(result) if result else None, "row_count": 1 if result else 0})
						elif fetch == "value":
							result = await connection.fetchval(query, *query_params, timeout=timeout)
							results.append({"value": result, "row_count": 1 if result is not None else 0})
						else:  # execute
							result = await connection.execute(query, *query_params, timeout=timeout)
							results.append({"status": result, "row_count": self._parse_execute_result(result)})
				
				return {"results": results, "transaction_successful": True}
			
			finally:
				self.active_connections -= 1
	
	async def _health_check(self) -> bool:
		"""Check PostgreSQL connection health."""
		try:
			async with self.connection_pool.acquire() as connection:
				result = await connection.fetchval("SELECT 1", timeout=5)
				return result == 1
		except Exception as e:
			logger.warning(self._log_connector_info(f"Health check failed: {e}"))
			return False
	
	def _parse_execute_result(self, result: str) -> int:
		"""Parse PostgreSQL execute result to get affected row count."""
		try:
			# Result format: "INSERT 0 1", "UPDATE 3", "DELETE 2", etc.
			parts = result.split()
			if len(parts) >= 2:
				return int(parts[-1])
			return 0
		except (ValueError, IndexError):
			return 0

class MongoDBAdapter(DatabaseConnector):
	"""High-performance MongoDB connector."""
	
	def __init__(self, config: MongoDBConfiguration):
		super().__init__(config)
		self.config: MongoDBConfiguration = config
		self.client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
		self.database = None
	
	async def _connect(self) -> None:
		"""Create MongoDB connection."""
		
		# Build connection URI
		uri = f"mongodb://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
		
		if self.config.auth_source != self.config.database:
			uri += f"?authSource={self.config.auth_source}"
		
		# Add additional parameters
		params = []
		if self.config.replica_set:
			params.append(f"replicaSet={self.config.replica_set}")
		if self.config.ssl_mode != "disable":
			params.append("ssl=true")
		
		if params:
			separator = "&" if "?" in uri else "?"
			uri += separator + "&".join(params)
		
		# Create client
		self.client = motor.motor_asyncio.AsyncIOMotorClient(
			uri,
			maxPoolSize=self.config.max_pool_size,
			minPoolSize=self.config.min_pool_size,
			maxIdleTimeMS=self.config.pool_recycle_time * 1000,
			serverSelectionTimeoutMS=self.config.connection_timeout * 1000
		)
		
		# Get database reference
		self.database = self.client[self.config.database]
		
		logger.info(self._log_connector_info("MongoDB connection established"))
	
	async def _disconnect(self) -> None:
		"""Close MongoDB connection."""
		if self.client:
			self.client.close()
			self.client = None
			self.database = None
		
		logger.info(self._log_connector_info("MongoDB connection closed"))
	
	async def _execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute MongoDB operation."""
		
		collection_name = parameters.get("collection")
		if not collection_name:
			raise ValueError("Collection name is required for MongoDB operations")
		
		collection = self.database[collection_name]
		
		if operation == "find":
			return await self._execute_find(collection, parameters)
		elif operation == "find_one":
			return await self._execute_find_one(collection, parameters)
		elif operation == "insert_one":
			return await self._execute_insert_one(collection, parameters)
		elif operation == "insert_many":
			return await self._execute_insert_many(collection, parameters)
		elif operation == "update_one":
			return await self._execute_update_one(collection, parameters)
		elif operation == "update_many":
			return await self._execute_update_many(collection, parameters)
		elif operation == "delete_one":
			return await self._execute_delete_one(collection, parameters)
		elif operation == "delete_many":
			return await self._execute_delete_many(collection, parameters)
		elif operation == "aggregate":
			return await self._execute_aggregate(collection, parameters)
		elif operation == "count":
			return await self._execute_count(collection, parameters)
		else:
			raise ValueError(f"Unsupported MongoDB operation: {operation}")
	
	async def _execute_find(self, collection, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute MongoDB find operation."""
		
		filter_query = params.get("filter", {})
		projection = params.get("projection")
		sort = params.get("sort")
		limit = params.get("limit")
		skip = params.get("skip")
		
		cursor = collection.find(filter_query, projection)
		
		if sort:
			cursor = cursor.sort(sort)
		if skip:
			cursor = cursor.skip(skip)
		if limit:
			cursor = cursor.limit(limit)
		
		documents = await cursor.to_list(length=limit)
		
		return {
			"documents": documents,
			"count": len(documents)
		}
	
	async def _execute_find_one(self, collection, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute MongoDB find_one operation."""
		
		filter_query = params.get("filter", {})
		projection = params.get("projection")
		
		document = await collection.find_one(filter_query, projection)
		
		return {
			"document": document,
			"found": document is not None
		}
	
	async def _execute_insert_one(self, collection, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute MongoDB insert_one operation."""
		
		document = params.get("document")
		if not document:
			raise ValueError("Document is required for insert_one operation")
		
		result = await collection.insert_one(document)
		
		return {
			"inserted_id": str(result.inserted_id),
			"acknowledged": result.acknowledged
		}
	
	async def _execute_insert_many(self, collection, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute MongoDB insert_many operation."""
		
		documents = params.get("documents")
		if not documents:
			raise ValueError("Documents are required for insert_many operation")
		
		result = await collection.insert_many(documents)
		
		return {
			"inserted_ids": [str(obj_id) for obj_id in result.inserted_ids],
			"inserted_count": len(result.inserted_ids),
			"acknowledged": result.acknowledged
		}
	
	async def _execute_update_one(self, collection, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute MongoDB update_one operation."""
		
		filter_query = params.get("filter", {})
		update = params.get("update")
		upsert = params.get("upsert", False)
		
		if not update:
			raise ValueError("Update document is required for update_one operation")
		
		result = await collection.update_one(filter_query, update, upsert=upsert)
		
		return {
			"matched_count": result.matched_count,
			"modified_count": result.modified_count,
			"upserted_id": str(result.upserted_id) if result.upserted_id else None,
			"acknowledged": result.acknowledged
		}
	
	async def _execute_update_many(self, collection, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute MongoDB update_many operation."""
		
		filter_query = params.get("filter", {})
		update = params.get("update")
		upsert = params.get("upsert", False)
		
		if not update:
			raise ValueError("Update document is required for update_many operation")
		
		result = await collection.update_many(filter_query, update, upsert=upsert)
		
		return {
			"matched_count": result.matched_count,
			"modified_count": result.modified_count,
			"upserted_id": str(result.upserted_id) if result.upserted_id else None,
			"acknowledged": result.acknowledged
		}
	
	async def _execute_delete_one(self, collection, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute MongoDB delete_one operation."""
		
		filter_query = params.get("filter", {})
		result = await collection.delete_one(filter_query)
		
		return {
			"deleted_count": result.deleted_count,
			"acknowledged": result.acknowledged
		}
	
	async def _execute_delete_many(self, collection, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute MongoDB delete_many operation."""
		
		filter_query = params.get("filter", {})
		result = await collection.delete_many(filter_query)
		
		return {
			"deleted_count": result.deleted_count,
			"acknowledged": result.acknowledged
		}
	
	async def _execute_aggregate(self, collection, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute MongoDB aggregate operation."""
		
		pipeline = params.get("pipeline", [])
		if not pipeline:
			raise ValueError("Pipeline is required for aggregate operation")
		
		cursor = collection.aggregate(pipeline)
		documents = await cursor.to_list(length=None)
		
		return {
			"documents": documents,
			"count": len(documents)
		}
	
	async def _execute_count(self, collection, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute MongoDB count operation."""
		
		filter_query = params.get("filter", {})
		count = await collection.count_documents(filter_query)
		
		return {
			"count": count
		}
	
	async def _health_check(self) -> bool:
		"""Check MongoDB connection health."""
		try:
			# Ping the database
			await self.client.admin.command("ping")
			return True
		except Exception as e:
			logger.warning(self._log_connector_info(f"Health check failed: {e}"))
			return False

# Export database connector classes
__all__ = [
	"DatabaseConnector",
	"DatabaseConfiguration", 
	"PostgreSQLAdapter",
	"PostgreSQLConfiguration",
	"MongoDBAdapter",
	"MongoDBConfiguration"
]