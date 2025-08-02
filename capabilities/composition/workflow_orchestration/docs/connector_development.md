# APG Workflow Orchestration - Connector Development Guide

**Complete guide for developing custom connectors and integrations**

© 2025 Datacraft. All rights reserved.

## Table of Contents

1. [Connector Architecture](#connector-architecture)
2. [Development Environment](#development-environment)
3. [Building Basic Connectors](#building-basic-connectors)
4. [Advanced Connector Features](#advanced-connector-features)
5. [APG Native Connectors](#apg-native-connectors)
6. [External System Connectors](#external-system-connectors)
7. [Testing Connectors](#testing-connectors)
8. [Security Implementation](#security-implementation)
9. [Performance Optimization](#performance-optimization)
10. [Deployment & Distribution](#deployment--distribution)
11. [Marketplace Guidelines](#marketplace-guidelines)
12. [Troubleshooting & Debugging](#troubleshooting--debugging)

## Connector Architecture

### Core Components

**Base Connector Interface:**
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
import asyncio
import logging

class ConnectorConfig(BaseModel):
	"""Base configuration for all connectors."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	name: str = Field(..., min_length=1, max_length=255)
	description: str | None = Field(None, max_length=1000)
	version: str = Field(default="1.0.0")
	timeout_seconds: int = Field(default=300, ge=1, le=3600)
	retry_attempts: int = Field(default=3, ge=0, le=10)
	retry_delay_seconds: int = Field(default=60, ge=1, le=300)
	cache_enabled: bool = Field(default=True)
	cache_ttl_seconds: int = Field(default=300, ge=0)

class ConnectorMetadata(BaseModel):
	"""Connector metadata for discovery and documentation."""
	model_config = ConfigDict(extra='forbid')
	
	connector_type: str = Field(..., min_length=1)
	display_name: str = Field(..., min_length=1)
	description: str = Field(..., min_length=1)
	category: str = Field(..., regex=r'^(database|api|file|messaging|cloud|apg|custom)$')
	tags: list[str] = Field(default_factory=list)
	
	# Technical specifications
	supported_operations: list[str] = Field(default_factory=list)
	input_schema: dict[str, Any] = Field(default_factory=dict)
	output_schema: dict[str, Any] = Field(default_factory=dict)
	configuration_schema: dict[str, Any] = Field(default_factory=dict)
	
	# Requirements and dependencies
	python_requirements: list[str] = Field(default_factory=list)
	system_requirements: list[str] = Field(default_factory=list)
	external_dependencies: list[str] = Field(default_factory=list)
	
	# Documentation
	documentation_url: str | None = None
	examples: list[dict] = Field(default_factory=list)
	
	# Versioning and maintenance
	version: str = Field(default="1.0.0")
	min_platform_version: str = Field(default="1.0.0")
	author: str = Field(..., min_length=1)
	license: str = Field(default="MIT")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)

class ExecutionResult(BaseModel):
	"""Result of connector execution."""
	model_config = ConfigDict(extra='forbid')
	
	success: bool
	output_data: Any = None
	error_message: str | None = None
	error_code: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	duration_ms: int | None = None
	resource_usage: dict[str, Any] = Field(default_factory=dict)

class BaseConnector(ABC):
	"""Base class for all workflow connectors."""
	
	def __init__(self, config: ConnectorConfig):
		self.config = config
		self.logger = logging.getLogger(f"{self.__class__.__name__}")
		self._cache = {}
		self._connection_pool = None
	
	@property
	@abstractmethod
	def metadata(self) -> ConnectorMetadata:
		"""Return connector metadata."""
		pass
	
	@abstractmethod
	async def execute(self, operation: str, input_data: Any, context: dict[str, Any]) -> ExecutionResult:
		"""Execute connector operation."""
		pass
	
	@abstractmethod
	async def validate_configuration(self) -> bool:
		"""Validate connector configuration."""
		pass
	
	@abstractmethod
	async def test_connection(self) -> bool:
		"""Test connector connectivity."""
		pass
	
	async def initialize(self):
		"""Initialize connector resources."""
		await self._setup_connection_pool()
		await self._validate_dependencies()
	
	async def cleanup(self):
		"""Cleanup connector resources."""
		if self._connection_pool:
			await self._connection_pool.close()
		self._cache.clear()
	
	async def _execute_with_retry(self, operation_func, *args, **kwargs) -> Any:
		"""Execute operation with retry logic."""
		last_error = None
		
		for attempt in range(self.config.retry_attempts + 1):
			try:
				return await operation_func(*args, **kwargs)
			except Exception as e:
				last_error = e
				self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
				
				if attempt < self.config.retry_attempts:
					await asyncio.sleep(self.config.retry_delay_seconds)
		
		raise last_error
	
	async def _cache_get(self, key: str) -> Any:
		"""Get value from cache."""
		if not self.config.cache_enabled:
			return None
		
		cached_item = self._cache.get(key)
		if cached_item and datetime.utcnow().timestamp() - cached_item['timestamp'] < self.config.cache_ttl_seconds:
			return cached_item['value']
		
		return None
	
	async def _cache_set(self, key: str, value: Any):
		"""Set value in cache."""
		if self.config.cache_enabled:
			self._cache[key] = {
				'value': value,
				'timestamp': datetime.utcnow().timestamp()
			}
```

### Connector Registry

**Dynamic Connector Registration:**
```python
class ConnectorRegistry:
	"""Registry for managing workflow connectors."""
	
	_connectors: dict[str, type[BaseConnector]] = {}
	_metadata_cache: dict[str, ConnectorMetadata] = {}
	
	@classmethod
	def register(cls, connector_class: type[BaseConnector]):
		"""Register a connector class."""
		# Create temporary instance to get metadata
		temp_config = ConnectorConfig(name="temp")
		temp_instance = connector_class(temp_config)
		metadata = temp_instance.metadata
		
		cls._connectors[metadata.connector_type] = connector_class
		cls._metadata_cache[metadata.connector_type] = metadata
		
		logging.info(f"Registered connector: {metadata.connector_type}")
	
	@classmethod
	def get_connector(cls, connector_type: str, config: ConnectorConfig) -> BaseConnector:
		"""Get connector instance."""
		if connector_type not in cls._connectors:
			raise ValueError(f"Unknown connector type: {connector_type}")
		
		connector_class = cls._connectors[connector_type]
		return connector_class(config)
	
	@classmethod
	def list_connectors(cls) -> list[ConnectorMetadata]:
		"""List all registered connectors."""
		return list(cls._metadata_cache.values())
	
	@classmethod
	def get_metadata(cls, connector_type: str) -> ConnectorMetadata:
		"""Get connector metadata."""
		if connector_type not in cls._metadata_cache:
			raise ValueError(f"Unknown connector type: {connector_type}")
		
		return cls._metadata_cache[connector_type]
	
	@classmethod
	def search_connectors(cls, category: str = None, tags: list[str] = None) -> list[ConnectorMetadata]:
		"""Search connectors by category and tags."""
		results = []
		
		for metadata in cls._metadata_cache.values():
			if category and metadata.category != category:
				continue
			
			if tags and not any(tag in metadata.tags for tag in tags):
				continue
			
			results.append(metadata)
		
		return results
```

## Development Environment

### Setup Development Environment

**Required Dependencies:**
```bash
# Python environment
python -m venv connector-dev
source connector-dev/bin/activate

# Core dependencies
pip install pydantic>=2.0.0
pip install asyncio-mqtt
pip install aiohttp
pip install sqlalchemy[asyncio]
pip install redis
```

**Development Tools:**
```bash
# Development and testing tools
pip install pytest
pip install pytest-asyncio  
pip install pytest-mock
pip install pytest-cov
pip install black
pip install isort
pip install mypy
```

**Project Structure:**
```
my_connector/
├── __init__.py
├── connector.py          # Main connector implementation
├── config.py            # Configuration models
├── exceptions.py        # Custom exceptions
├── utils.py            # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_connector.py
│   ├── test_integration.py
│   └── fixtures/
├── examples/
│   ├── basic_usage.py
│   └── advanced_usage.py
├── docs/
│   ├── README.md
│   └── configuration.md
├── requirements.txt
├── setup.py
└── pyproject.toml
```

## Building Basic Connectors

### HTTP API Connector

**Complete HTTP API Connector Implementation:**
```python
from typing import Any, Dict, Optional
import aiohttp
import json
from urllib.parse import urljoin

class HTTPConnectorConfig(ConnectorConfig):
	"""Configuration for HTTP API connector."""
	
	base_url: str = Field(..., description="Base URL for the API")
	default_headers: dict[str, str] = Field(default_factory=dict)
	auth_type: str = Field(default="none", regex=r'^(none|basic|bearer|api_key|oauth2)$')
	auth_config: dict[str, Any] = Field(default_factory=dict)
	verify_ssl: bool = Field(default=True)
	connection_timeout: int = Field(default=30, ge=1)
	read_timeout: int = Field(default=300, ge=1)

class HTTPConnector(BaseConnector):
	"""HTTP API connector for RESTful services."""
	
	def __init__(self, config: HTTPConnectorConfig):
		super().__init__(config)
		self.config: HTTPConnectorConfig = config
		self._session: Optional[aiohttp.ClientSession] = None
	
	@property
	def metadata(self) -> ConnectorMetadata:
		return ConnectorMetadata(
			connector_type="http_api",
			display_name="HTTP API Connector",
			description="Connect to RESTful APIs and web services",
			category="api",
			tags=["http", "rest", "api", "web"],
			supported_operations=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
			input_schema={
				"type": "object",
				"properties": {
					"method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]},
					"endpoint": {"type": "string"},
					"headers": {"type": "object"},
					"params": {"type": "object"},
					"data": {"type": ["object", "string", "null"]},
					"json": {"type": ["object", "array", "null"]}
				},
				"required": ["method", "endpoint"]
			},
			output_schema={
				"type": "object",
				"properties": {
					"status_code": {"type": "integer"},
					"headers": {"type": "object"},
					"data": {"type": ["object", "array", "string", "null"]},
					"url": {"type": "string"}
				}
			},
			configuration_schema={
				"type": "object",
				"properties": {
					"base_url": {"type": "string", "format": "uri"},
					"auth_type": {"type": "string", "enum": ["none", "basic", "bearer", "api_key", "oauth2"]},
					"auth_config": {"type": "object"},
					"default_headers": {"type": "object"},
					"verify_ssl": {"type": "boolean"},
					"connection_timeout": {"type": "integer"},
					"read_timeout": {"type": "integer"}
				},
				"required": ["base_url"]
			},
			python_requirements=["aiohttp>=3.8.0"],
			author="APG Development Team",
			version="1.0.0"
		)
	
	async def initialize(self):
		"""Initialize HTTP session."""
		await super().initialize()
		
		connector = aiohttp.TCPConnector(
			verify_ssl=self.config.verify_ssl,
			limit=100,
			limit_per_host=20
		)
		
		timeout = aiohttp.ClientTimeout(
			connect=self.config.connection_timeout,
			total=self.config.read_timeout
		)
		
		self._session = aiohttp.ClientSession(
			connector=connector,
			timeout=timeout,
			headers=self.config.default_headers
		)
	
	async def cleanup(self):
		"""Cleanup HTTP session."""
		if self._session:
			await self._session.close()
		await super().cleanup()
	
	async def execute(self, operation: str, input_data: Any, context: dict[str, Any]) -> ExecutionResult:
		"""Execute HTTP request."""
		start_time = datetime.utcnow()
		
		try:
			# Validate input
			if not isinstance(input_data, dict):
				raise ValueError("Input data must be a dictionary")
			
			method = input_data.get("method", operation).upper()
			endpoint = input_data["endpoint"]
			
			# Build request parameters
			url = urljoin(self.config.base_url, endpoint)
			headers = {**self.config.default_headers, **input_data.get("headers", {})}
			params = input_data.get("params", {})
			
			# Add authentication
			headers = await self._add_authentication(headers)
			
			# Prepare request data
			request_kwargs = {
				"headers": headers,
				"params": params
			}
			
			if "data" in input_data:
				request_kwargs["data"] = input_data["data"]
			elif "json" in input_data:
				request_kwargs["json"] = input_data["json"]
			
			# Execute request with retry logic
			response_data = await self._execute_with_retry(
				self._make_request, method, url, **request_kwargs
			)
			
			duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
			
			return ExecutionResult(
				success=True,
				output_data=response_data,
				metadata={
					"method": method,
					"url": url,
					"request_headers": dict(headers),
					"response_headers": response_data.get("headers", {})
				},
				duration_ms=duration_ms
			)
			
		except Exception as e:
			duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
			
			return ExecutionResult(
				success=False,
				error_message=str(e),
				error_code=type(e).__name__,
				duration_ms=duration_ms
			)
	
	async def _make_request(self, method: str, url: str, **kwargs) -> dict:
		"""Make HTTP request."""
		if not self._session:
			raise RuntimeError("HTTP session not initialized")
		
		async with self._session.request(method, url, **kwargs) as response:
			# Parse response data
			content_type = response.headers.get("content-type", "").lower()
			
			if "application/json" in content_type:
				data = await response.json()
			elif "text/" in content_type or "application/xml" in content_type:
				data = await response.text()
			else:
				data = await response.read()
			
			return {
				"status_code": response.status,
				"headers": dict(response.headers),
				"data": data,
				"url": str(response.url)
			}
	
	async def _add_authentication(self, headers: dict) -> dict:
		"""Add authentication to request headers."""
		auth_headers = headers.copy()
		
		if self.config.auth_type == "bearer":
			token = self.config.auth_config.get("token")
			if token:
				auth_headers["Authorization"] = f"Bearer {token}"
		
		elif self.config.auth_type == "api_key":
			api_key = self.config.auth_config.get("api_key")
			header_name = self.config.auth_config.get("header_name", "X-API-Key")
			if api_key:
				auth_headers[header_name] = api_key
		
		elif self.config.auth_type == "basic":
			username = self.config.auth_config.get("username")
			password = self.config.auth_config.get("password")
			if username and password:
				import base64
				credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
				auth_headers["Authorization"] = f"Basic {credentials}"
		
		return auth_headers
	
	async def validate_configuration(self) -> bool:
		"""Validate HTTP connector configuration."""
		try:
			# Validate base URL
			if not self.config.base_url.startswith(("http://", "https://")):
				raise ValueError("base_url must start with http:// or https://")
			
			# Validate authentication configuration
			if self.config.auth_type != "none":
				required_fields = {
					"bearer": ["token"],
					"api_key": ["api_key"],
					"basic": ["username", "password"],
					"oauth2": ["client_id", "client_secret"]
				}
				
				required = required_fields.get(self.config.auth_type, [])
				for field in required:
					if field not in self.config.auth_config:
						raise ValueError(f"Missing required auth field: {field}")
			
			return True
			
		except Exception as e:
			self.logger.error(f"Configuration validation failed: {e}")
			return False
	
	async def test_connection(self) -> bool:
		"""Test HTTP connectivity."""
		try:
			if not self._session:
				await self.initialize()
			
			# Make a simple HEAD or GET request to test connectivity
			test_url = self.config.base_url
			headers = await self._add_authentication(self.config.default_headers)
			
			async with self._session.head(test_url, headers=headers) as response:
				return 200 <= response.status < 500  # Accept all non-server-error responses
				
		except Exception as e:
			self.logger.error(f"Connection test failed: {e}")
			return False

# Register the connector
ConnectorRegistry.register(HTTPConnector)
```

### Database Connector

**Complete Database Connector Implementation:**
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import pandas as pd

class DatabaseConnectorConfig(ConnectorConfig):
	"""Configuration for database connector."""
	
	connection_string: str = Field(..., description="Database connection string")
	pool_size: int = Field(default=10, ge=1, le=100)
	max_overflow: int = Field(default=20, ge=0, le=100)
	pool_timeout: int = Field(default=30, ge=1)
	query_timeout: int = Field(default=300, ge=1)
	isolation_level: str = Field(default="READ_COMMITTED")

class DatabaseConnector(BaseConnector):
	"""Database connector for SQL databases."""
	
	def __init__(self, config: DatabaseConnectorConfig):
		super().__init__(config)
		self.config: DatabaseConnectorConfig = config
		self._engine = None
		self._session_factory = None
	
	@property
	def metadata(self) -> ConnectorMetadata:
		return ConnectorMetadata(
			connector_type="database",
			display_name="Database Connector",
			description="Connect to SQL databases (PostgreSQL, MySQL, SQLite, etc.)",
			category="database",
			tags=["database", "sql", "postgresql", "mysql", "sqlite"],
			supported_operations=["query", "execute", "bulk_insert", "bulk_update", "transaction"],
			input_schema={
				"type": "object",
				"properties": {
					"sql": {"type": "string"},
					"parameters": {"type": ["object", "array", "null"]},
					"fetch_mode": {"type": "string", "enum": ["all", "one", "many"]},
					"chunk_size": {"type": "integer", "minimum": 1}
				},
				"required": ["sql"]
			},
			output_schema={
				"type": "object",
				"properties": {
					"rows": {"type": "array"},
					"row_count": {"type": "integer"},
					"columns": {"type": "array"},
					"execution_time_ms": {"type": "integer"}
				}
			},
			python_requirements=["sqlalchemy[asyncio]>=2.0.0", "pandas>=1.5.0"],
			author="APG Development Team",
			version="1.0.0"
		)
	
	async def initialize(self):
		"""Initialize database engine and session factory."""
		await super().initialize()
		
		self._engine = create_async_engine(
			self.config.connection_string,
			pool_size=self.config.pool_size,
			max_overflow=self.config.max_overflow,
			pool_timeout=self.config.pool_timeout,
			echo=False  # Set to True for SQL debugging
		)
		
		self._session_factory = sessionmaker(
			bind=self._engine,
			class_=AsyncSession,
			expire_on_commit=False
		)
	
	async def cleanup(self):
		"""Cleanup database connections."""
		if self._engine:
			await self._engine.dispose()
		await super().cleanup()
	
	async def execute(self, operation: str, input_data: Any, context: dict[str, Any]) -> ExecutionResult:
		"""Execute database operation."""
		start_time = datetime.utcnow()
		
		try:
			if not isinstance(input_data, dict):
				raise ValueError("Input data must be a dictionary")
			
			sql = input_data["sql"]
			parameters = input_data.get("parameters")
			fetch_mode = input_data.get("fetch_mode", "all")
			
			if operation == "query":
				result_data = await self._execute_query(sql, parameters, fetch_mode)
			elif operation == "execute":
				result_data = await self._execute_statement(sql, parameters)
			elif operation == "bulk_insert":
				result_data = await self._bulk_insert(input_data)
			elif operation == "transaction":
				result_data = await self._execute_transaction(input_data)
			else:
				raise ValueError(f"Unsupported operation: {operation}")
			
			duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
			result_data["execution_time_ms"] = duration_ms
			
			return ExecutionResult(
				success=True,
				output_data=result_data,
				metadata={
					"operation": operation,
					"sql": sql[:100] + "..." if len(sql) > 100 else sql,
					"parameter_count": len(parameters) if parameters else 0
				},
				duration_ms=duration_ms
			)
			
		except Exception as e:
			duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
			
			return ExecutionResult(
				success=False,
				error_message=str(e),
				error_code=type(e).__name__,
				duration_ms=duration_ms
			)
	
	async def _execute_query(self, sql: str, parameters: Any, fetch_mode: str) -> dict:
		"""Execute SELECT query."""
		async with self._session_factory() as session:
			result = await session.execute(text(sql), parameters)
			
			if fetch_mode == "one":
				row = result.fetchone()
				rows = [dict(row._mapping)] if row else []
			elif fetch_mode == "many":
				chunk_size = parameters.get("chunk_size", 1000) if isinstance(parameters, dict) else 1000
				rows = [dict(row._mapping) for row in result.fetchmany(chunk_size)]
			else:  # all
				rows = [dict(row._mapping) for row in result.fetchall()]
			
			columns = list(result.keys()) if rows else []
			
			return {
				"rows": rows,
				"row_count": len(rows),
				"columns": columns
			}
	
	async def _execute_statement(self, sql: str, parameters: Any) -> dict:
		"""Execute INSERT/UPDATE/DELETE statement."""
		async with self._session_factory() as session:
			result = await session.execute(text(sql), parameters)
			await session.commit()
			
			return {
				"rows_affected": result.rowcount,
				"row_count": result.rowcount,
				"columns": []
			}
	
	async def _bulk_insert(self, input_data: dict) -> dict:
		"""Execute bulk insert operation."""
		table_name = input_data["table_name"]
		data = input_data["data"]
		
		if not isinstance(data, list):
			raise ValueError("Bulk insert data must be a list of dictionaries")
		
		# Convert to DataFrame for efficient bulk operations
		df = pd.DataFrame(data)
		
		# Use pandas to_sql for efficient bulk insert
		await df.to_sql(
			table_name,
			self._engine,
			if_exists="append",
			index=False,
			method="multi"
		)
		
		return {
			"rows_affected": len(data),
			"row_count": len(data),
			"columns": list(df.columns)
		}
	
	async def _execute_transaction(self, input_data: dict) -> dict:
		"""Execute multiple statements in a transaction."""
		statements = input_data["statements"]
		
		if not isinstance(statements, list):
			raise ValueError("Transaction statements must be a list")
		
		async with self._session_factory() as session:
			async with session.begin():
				results = []
				total_affected = 0
				
				for stmt_data in statements:
					sql = stmt_data["sql"]
					parameters = stmt_data.get("parameters")
					
					result = await session.execute(text(sql), parameters)
					results.append({
						"sql": sql[:50] + "..." if len(sql) > 50 else sql,
						"rows_affected": result.rowcount
					})
					total_affected += result.rowcount
			
			return {
				"transaction_results": results,
				"total_rows_affected": total_affected,
				"row_count": total_affected,
				"columns": []
			}
	
	async def validate_configuration(self) -> bool:
		"""Validate database configuration."""
		try:
			# Basic connection string validation
			if not self.config.connection_string:
				raise ValueError("Connection string is required")
			
			# Test engine creation
			test_engine = create_async_engine(
				self.config.connection_string,
				pool_size=1,
				max_overflow=0
			)
			
			# Test connection
			async with test_engine.connect() as conn:
				await conn.execute(text("SELECT 1"))
			
			await test_engine.dispose()
			return True
			
		except Exception as e:
			self.logger.error(f"Database configuration validation failed: {e}")
			return False
	
	async def test_connection(self) -> bool:
		"""Test database connectivity."""
		try:
			if not self._engine:
				await self.initialize()
			
			async with self._engine.connect() as conn:
				await conn.execute(text("SELECT 1"))
			
			return True
			
		except Exception as e:
			self.logger.error(f"Database connection test failed: {e}")
			return False

# Register the connector
ConnectorRegistry.register(DatabaseConnector)
```

## Advanced Connector Features

### Streaming Data Connector

**Real-time Data Streaming:**
```python
import asyncio
from asyncio import Queue
from typing import AsyncIterator

class StreamingConnectorConfig(ConnectorConfig):
	"""Configuration for streaming data connector."""
	
	stream_source: str = Field(..., description="Stream source identifier")
	batch_size: int = Field(default=100, ge=1, le=10000)
	buffer_size: int = Field(default=1000, ge=1)
	stream_timeout: int = Field(default=30, ge=1)
	auto_commit: bool = Field(default=True)
	checkpoint_interval: int = Field(default=1000, ge=1)

class StreamingConnector(BaseConnector):
	"""Connector for streaming data processing."""
	
	def __init__(self, config: StreamingConnectorConfig):
		super().__init__(config)
		self.config: StreamingConnectorConfig = config
		self._stream_queue: Queue = None
		self._consumer_task: asyncio.Task = None
		self._is_streaming = False
	
	@property
	def metadata(self) -> ConnectorMetadata:
		return ConnectorMetadata(
			connector_type="streaming",
			display_name="Streaming Data Connector",
			description="Process streaming data in real-time",
			category="messaging",
			tags=["streaming", "realtime", "kafka", "rabbitmq", "redis"],
			supported_operations=["consume", "produce", "stream_process"],
			author="APG Development Team",
			version="1.0.0"
		)
	
	async def execute(self, operation: str, input_data: Any, context: dict[str, Any]) -> ExecutionResult:
		"""Execute streaming operation."""
		if operation == "consume":
			return await self._consume_stream(input_data, context)
		elif operation == "produce":
			return await self._produce_to_stream(input_data, context)
		elif operation == "stream_process":
			return await self._process_stream(input_data, context)
		else:
			raise ValueError(f"Unsupported streaming operation: {operation}")
	
	async def _consume_stream(self, input_data: dict, context: dict) -> ExecutionResult:
		"""Consume messages from stream."""
		messages = []
		checkpoint_count = 0
		
		try:
			async for message in self._stream_iterator():
				messages.append(message)
				checkpoint_count += 1
				
				# Process in batches
				if len(messages) >= self.config.batch_size:
					await self._process_batch(messages, context)
					messages = []
				
				# Checkpoint progress
				if checkpoint_count >= self.config.checkpoint_interval:
					await self._create_checkpoint(message)
					checkpoint_count = 0
				
				# Yield control to allow other operations
				await asyncio.sleep(0)
			
			# Process remaining messages
			if messages:
				await self._process_batch(messages, context)
			
			return ExecutionResult(
				success=True,
				output_data={"messages_processed": checkpoint_count},
				metadata={"operation": "consume", "batch_size": self.config.batch_size}
			)
			
		except Exception as e:
			return ExecutionResult(
				success=False,
				error_message=str(e),
				error_code=type(e).__name__
			)
	
	async def _stream_iterator(self) -> AsyncIterator[dict]:
		"""Create async iterator for stream messages."""
		while self._is_streaming:
			try:
				# Get message from queue with timeout
				message = await asyncio.wait_for(
					self._stream_queue.get(),
					timeout=self.config.stream_timeout
				)
				
				if message is None:  # Termination signal
					break
				
				yield message
				
				# Mark task as done
				self._stream_queue.task_done()
				
			except asyncio.TimeoutError:
				# No messages available, continue
				continue
			except Exception as e:
				self.logger.error(f"Stream iteration error: {e}")
				break
	
	async def _process_batch(self, messages: list, context: dict):
		"""Process batch of messages."""
		# Implement batch processing logic
		for message in messages:
			# Process individual message
			await self._process_message(message, context)
	
	async def _process_message(self, message: dict, context: dict):
		"""Process individual message."""
		# Implement message processing logic
		self.logger.debug(f"Processing message: {message.get('id', 'unknown')}")
	
	async def _create_checkpoint(self, message: dict):
		"""Create processing checkpoint."""
		checkpoint_data = {
			"timestamp": datetime.utcnow().isoformat(),
			"message_id": message.get("id"),
			"offset": message.get("offset")
		}
		
		# Store checkpoint (implement storage logic)
		await self._store_checkpoint(checkpoint_data)
	
	async def _store_checkpoint(self, checkpoint_data: dict):
		"""Store checkpoint data."""
		# Implement checkpoint storage logic
		pass
```

### Event-Driven Connector

**Event-Driven Processing:**
```python
from typing import Callable, Dict, List
import uuid

class EventDrivenConnectorConfig(ConnectorConfig):
	"""Configuration for event-driven connector."""
	
	event_source: str = Field(..., description="Event source identifier")
	event_filters: list[dict] = Field(default_factory=list)
	dead_letter_queue: bool = Field(default=True)
	max_retry_attempts: int = Field(default=3, ge=0)
	event_ordering: bool = Field(default=False)

class EventDrivenConnector(BaseConnector):
	"""Event-driven connector for reactive processing."""
	
	def __init__(self, config: EventDrivenConnectorConfig):
		super().__init__(config)
		self.config: EventDrivenConnectorConfig = config
		self._event_handlers: Dict[str, Callable] = {}
		self._event_filters: List[Callable] = []
		self._dead_letter_queue: List[dict] = []
	
	@property
	def metadata(self) -> ConnectorMetadata:
		return ConnectorMetadata(
			connector_type="event_driven",
			display_name="Event-Driven Connector",
			description="Process events in reactive manner",
			category="messaging",
			tags=["events", "reactive", "publish-subscribe", "messaging"],
			supported_operations=["subscribe", "publish", "handle_event"],
			author="APG Development Team",
			version="1.0.0"
		)
	
	def register_event_handler(self, event_type: str, handler: Callable):
		"""Register event handler for specific event type."""
		self._event_handlers[event_type] = handler
		self.logger.info(f"Registered handler for event type: {event_type}")
	
	def add_event_filter(self, filter_func: Callable[[dict], bool]):
		"""Add event filter function."""
		self._event_filters.append(filter_func)
	
	async def execute(self, operation: str, input_data: Any, context: dict[str, Any]) -> ExecutionResult:
		"""Execute event-driven operation."""
		if operation == "subscribe":
			return await self._subscribe_to_events(input_data, context)
		elif operation == "publish":
			return await self._publish_event(input_data, context)
		elif operation == "handle_event":
			return await self._handle_single_event(input_data, context)
		else:
			raise ValueError(f"Unsupported event operation: {operation}")
	
	async def _subscribe_to_events(self, input_data: dict, context: dict) -> ExecutionResult:
		"""Subscribe to event stream."""
		event_types = input_data.get("event_types", [])
		duration = input_data.get("duration_seconds", 60)
		
		processed_events = 0
		failed_events = 0
		
		try:
			# Start event subscription
			end_time = datetime.utcnow().timestamp() + duration
			
			while datetime.utcnow().timestamp() < end_time:
				# Get events from source (implement source-specific logic)
				events = await self._fetch_events(event_types)
				
				for event in events:
					try:
						# Apply filters
						if not await self._passes_filters(event):
							continue
						
						# Process event
						await self._process_event(event, context)
						processed_events += 1
						
					except Exception as e:
						failed_events += 1
						await self._handle_failed_event(event, e)
				
				# Small delay to prevent tight loop
				await asyncio.sleep(0.1)
			
			return ExecutionResult(
				success=True,
				output_data={
					"processed_events": processed_events,
					"failed_events": failed_events,
					"dead_letter_count": len(self._dead_letter_queue)
				},
				metadata={"operation": "subscribe", "duration": duration}
			)
			
		except Exception as e:
			return ExecutionResult(
				success=False,
				error_message=str(e),
				error_code=type(e).__name__
			)
	
	async def _fetch_events(self, event_types: list) -> list[dict]:
		"""Fetch events from source."""
		# Implement source-specific event fetching
		# This is a placeholder implementation
		return []
	
	async def _passes_filters(self, event: dict) -> bool:
		"""Check if event passes all filters."""
		for filter_func in self._event_filters:
			if not filter_func(event):
				return False
		return True
	
	async def _process_event(self, event: dict, context: dict):
		"""Process single event."""
		event_type = event.get("type", "unknown")
		
		if event_type in self._event_handlers:
			handler = self._event_handlers[event_type]
			await handler(event, context)
		else:
			self.logger.warning(f"No handler registered for event type: {event_type}")
	
	async def _handle_failed_event(self, event: dict, error: Exception):
		"""Handle failed event processing."""
		retry_count = event.get("retry_count", 0)
		
		if retry_count < self.config.max_retry_attempts:
			# Retry the event
			event["retry_count"] = retry_count + 1
			await self._retry_event(event)
		else:
			# Send to dead letter queue
			if self.config.dead_letter_queue:
				dead_letter_event = {
					"event": event,
					"error": str(error),
					"failed_at": datetime.utcnow().isoformat(),
					"retry_count": retry_count
				}
				self._dead_letter_queue.append(dead_letter_event)
	
	async def _retry_event(self, event: dict):
		"""Retry failed event processing."""
		try:
			await self._process_event(event, {})
		except Exception as e:
			await self._handle_failed_event(event, e)
	
	async def validate_configuration(self) -> bool:
		"""Validate event-driven configuration."""
		try:
			if not self.config.event_source:
				raise ValueError("Event source is required")
			
			# Validate event filters
			for filter_config in self.config.event_filters:
				if not isinstance(filter_config, dict):
					raise ValueError("Event filters must be dictionaries")
			
			return True
			
		except Exception as e:
			self.logger.error(f"Event-driven configuration validation failed: {e}")
			return False
	
	async def test_connection(self) -> bool:
		"""Test event source connectivity."""
		try:
			# Test connection to event source
			test_events = await self._fetch_events([])
			return True
			
		except Exception as e:
			self.logger.error(f"Event source connection test failed: {e}")
			return False
```

## APG Native Connectors

### User Management Connector

**APG User Management Integration:**
```python
class APGUserManagementConfig(ConnectorConfig):
	"""Configuration for APG User Management connector."""
	
	apg_base_url: str = Field(..., description="APG platform base URL")
	api_key: str = Field(..., description="APG API key")
	tenant_id: str | None = Field(None, description="Tenant ID for multi-tenant deployments")
	user_fields: list[str] = Field(default_factory=lambda: ["id", "username", "email", "roles"])

class APGUserManagementConnector(BaseConnector):
	"""Connector for APG User Management capability."""
	
	def __init__(self, config: APGUserManagementConfig):
		super().__init__(config)
		self.config: APGUserManagementConfig = config
		self._apg_client = None
	
	@property
	def metadata(self) -> ConnectorMetadata:
		return ConnectorMetadata(
			connector_type="apg_user_management",
			display_name="APG User Management",
			description="Integrate with APG User Management capability",
			category="apg",
			tags=["apg", "users", "authentication", "authorization"],
			supported_operations=[
				"create_user", "get_user", "update_user", "delete_user",
				"list_users", "authenticate_user", "assign_role", "remove_role"
			],
			input_schema={
				"type": "object",
				"properties": {
					"user_data": {"type": "object"},
					"user_id": {"type": "string"},
					"username": {"type": "string"},
					"email": {"type": "string"},
					"role": {"type": "string"},
					"filters": {"type": "object"}
				}
			},
			output_schema={
				"type": "object",
				"properties": {
					"user": {"type": "object"},
					"users": {"type": "array"},
					"success": {"type": "boolean"},
					"message": {"type": "string"}
				}
			},
			python_requirements=["apg-sdk>=1.0.0", "httpx>=0.24.0"],
			author="APG Development Team",
			version="1.0.0"
		)
	
	async def initialize(self):
		"""Initialize APG client."""
		await super().initialize()
		
		from apg.client import APGClient
		
		self._apg_client = APGClient(
			base_url=self.config.apg_base_url,
			api_key=self.config.api_key,
			tenant_id=self.config.tenant_id
		)
	
	async def execute(self, operation: str, input_data: Any, context: dict[str, Any]) -> ExecutionResult:
		"""Execute user management operation."""
		start_time = datetime.utcnow()
		
		try:
			if operation == "create_user":
				result_data = await self._create_user(input_data)
			elif operation == "get_user":
				result_data = await self._get_user(input_data)
			elif operation == "update_user":
				result_data = await self._update_user(input_data)
			elif operation == "delete_user":
				result_data = await self._delete_user(input_data)
			elif operation == "list_users":
				result_data = await self._list_users(input_data)
			elif operation == "authenticate_user":
				result_data = await self._authenticate_user(input_data)
			elif operation == "assign_role":
				result_data = await self._assign_role(input_data)
			elif operation == "remove_role":
				result_data = await self._remove_role(input_data)
			else:
				raise ValueError(f"Unsupported user management operation: {operation}")
			
			duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
			
			return ExecutionResult(
				success=True,
				output_data=result_data,
				metadata={"operation": operation, "apg_capability": "user_management"},
				duration_ms=duration_ms
			)
			
		except Exception as e:
			duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
			
			return ExecutionResult(
				success=False,
				error_message=str(e),
				error_code=type(e).__name__,
				duration_ms=duration_ms
			)
	
	async def _create_user(self, input_data: dict) -> dict:
		"""Create new user."""
		user_data = input_data["user_data"]
		
		required_fields = ["username", "email", "password"]
		for field in required_fields:
			if field not in user_data:
				raise ValueError(f"Missing required field: {field}")
		
		response = await self._apg_client.user_management.create_user(
			username=user_data["username"],
			email=user_data["email"],
			password=user_data["password"],
			first_name=user_data.get("first_name"),
			last_name=user_data.get("last_name"),
			metadata=user_data.get("metadata", {})
		)
		
		return {
			"user": self._filter_user_fields(response.data),
			"success": response.success,
			"message": "User created successfully" if response.success else response.error_message
		}
	
	async def _get_user(self, input_data: dict) -> dict:
		"""Get user by ID or username."""
		if "user_id" in input_data:
			response = await self._apg_client.user_management.get_user_by_id(input_data["user_id"])
		elif "username" in input_data:
			response = await self._apg_client.user_management.get_user_by_username(input_data["username"])
		elif "email" in input_data:
			response = await self._apg_client.user_management.get_user_by_email(input_data["email"])
		else:
			raise ValueError("Must provide user_id, username, or email")
		
		return {
			"user": self._filter_user_fields(response.data) if response.success else None,
			"success": response.success,
			"message": "User retrieved successfully" if response.success else response.error_message
		}
	
	async def _update_user(self, input_data: dict) -> dict:
		"""Update user information."""
		user_id = input_data["user_id"]
		user_data = input_data["user_data"]
		
		response = await self._apg_client.user_management.update_user(
			user_id=user_id,
			**user_data
		)
		
		return {
			"user": self._filter_user_fields(response.data) if response.success else None,
			"success": response.success,
			"message": "User updated successfully" if response.success else response.error_message
		}
	
	async def _delete_user(self, input_data: dict) -> dict:
		"""Delete user."""
		user_id = input_data["user_id"]
		
		response = await self._apg_client.user_management.delete_user(user_id)
		
		return {
			"success": response.success,
			"message": "User deleted successfully" if response.success else response.error_message
		}
	
	async def _list_users(self, input_data: dict) -> dict:
		"""List users with optional filters."""
		filters = input_data.get("filters", {})
		limit = filters.get("limit", 50)
		offset = filters.get("offset", 0)
		
		response = await self._apg_client.user_management.list_users(
			limit=limit,
			offset=offset,
			filters=filters
		)
		
		users = [self._filter_user_fields(user) for user in response.data] if response.success else []
		
		return {
			"users": users,
			"total_count": len(users),
			"limit": limit,
			"offset": offset,
			"success": response.success,
			"message": f"Retrieved {len(users)} users" if response.success else response.error_message
		}
	
	async def _authenticate_user(self, input_data: dict) -> dict:
		"""Authenticate user credentials."""
		username = input_data["username"]
		password = input_data["password"]
		
		response = await self._apg_client.user_management.authenticate_user(
			username=username,
			password=password
		)
		
		return {
			"authenticated": response.success,
			"user": self._filter_user_fields(response.data) if response.success else None,
			"token": response.data.get("access_token") if response.success else None,
			"success": response.success,
			"message": "Authentication successful" if response.success else "Authentication failed"
		}
	
	async def _assign_role(self, input_data: dict) -> dict:
		"""Assign role to user."""
		user_id = input_data["user_id"]
		role = input_data["role"]
		
		response = await self._apg_client.user_management.assign_role(
			user_id=user_id,
			role=role
		)
		
		return {
			"success": response.success,
			"message": f"Role '{role}' assigned successfully" if response.success else response.error_message
		}
	
	async def _remove_role(self, input_data: dict) -> dict:
		"""Remove role from user."""
		user_id = input_data["user_id"]
		role = input_data["role"]
		
		response = await self._apg_client.user_management.remove_role(
			user_id=user_id,
			role=role
		)
		
		return {
			"success": response.success,
			"message": f"Role '{role}' removed successfully" if response.success else response.error_message
		}
	
	def _filter_user_fields(self, user_data: dict) -> dict:
		"""Filter user data to include only configured fields."""
		if not user_data:
			return {}
		
		filtered_data = {}
		for field in self.config.user_fields:
			if field in user_data:
				filtered_data[field] = user_data[field]
		
		return filtered_data
	
	async def validate_configuration(self) -> bool:
		"""Validate APG User Management configuration."""
		try:
			if not self.config.apg_base_url:
				raise ValueError("APG base URL is required")
			
			if not self.config.api_key:
				raise ValueError("APG API key is required")
			
			if not self.config.apg_base_url.startswith(("http://", "https://")):
				raise ValueError("APG base URL must be a valid HTTP/HTTPS URL")
			
			return True
			
		except Exception as e:
			self.logger.error(f"APG User Management configuration validation failed: {e}")
			return False
	
	async def test_connection(self) -> bool:
		"""Test APG User Management connectivity."""
		try:
			if not self._apg_client:
				await self.initialize()
			
			# Test connection by attempting to list users with limit 1
			response = await self._apg_client.user_management.list_users(limit=1)
			return response.success
			
		except Exception as e:
			self.logger.error(f"APG User Management connection test failed: {e}")
			return False

# Register the connector
ConnectorRegistry.register(APGUserManagementConnector)
```

This connector development guide provides comprehensive coverage for building enterprise-grade connectors. Continue with the remaining sections to complete Phase 11.2: Advanced Documentation.