#!/usr/bin/env python3
"""
APG Metadata Management - API Connectors
Connectors for discovering metadata from API-based data sources

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from urllib.parse import urlparse, urljoin

try:
	import httpx
except ImportError:
	httpx = None

try:
	from kafka import KafkaConsumer, KafkaAdminClient
	from kafka.structs import TopicPartition
	from kafka.errors import KafkaError
except ImportError:
	KafkaConsumer = None
	KafkaAdminClient = None

from .base_connector import (
	BaseConnector, ConnectorConfig, DiscoveryResult, AssetMetadata,
	ColumnMetadata, ConnectorType, DataType, should_include_asset
)


class RESTAPIConnector(BaseConnector):
	"""REST API metadata discovery connector"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.API
		self.source_system = "rest_api"
		self.client = None
		self.base_url = config.connection_string
		self.discovered_endpoints = []
		self.auth_headers = {}
	
	async def connect(self) -> bool:
		"""Establish connection to REST API"""
		try:
			if httpx is None:
				raise ImportError("httpx library is required for REST API connector")
			
			# Setup authentication headers
			if self.config.username and self.config.password:
				import base64
				creds = base64.b64encode(f"{self.config.username}:{self.config.password}".encode()).decode()
				self.auth_headers["Authorization"] = f"Basic {creds}"
			
			# Add any additional headers from config
			self.auth_headers.update(self.config.additional_params.get("headers", {}))
			
			# Create HTTP client
			self.client = httpx.AsyncClient(
				headers=self.auth_headers,
				timeout=self.config.connection_timeout,
				verify=self.config.use_ssl
			)
			
			self.is_connected = True
			return True
		except Exception as e:
			await self._log_error(f"Failed to connect to REST API: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Close connection to REST API"""
		if self.client:
			await self.client.aclose()
			self.client = None
		self.is_connected = False
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test connection to REST API"""
		if not self.is_connected:
			if not await self.connect():
				return {"status": "error", "message": "Failed to connect to REST API"}
		
		try:
			# Try to make a basic request to the root endpoint
			response = await self.client.get(self.base_url)
			return {
				"status": "success",
				"message": "Connected to REST API successfully",
				"response_code": response.status_code,
				"url": str(response.url)
			}
		except Exception as e:
			return {"status": "error", "message": f"Connection test failed: {str(e)}"}
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover REST API endpoints and their metadata"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		
		if not self.is_connected:
			if not await self.connect():
				result.add_error("Failed to connect to REST API")
				result.complete_discovery()
				return result
		
		try:
			# Try to discover endpoints through various methods
			await self._discover_openapi_endpoints(result)
			await self._discover_common_endpoints(result)
			await self._discover_from_links(result)
			
		except Exception as e:
			result.add_error(f"Error during asset discovery: {str(e)}")
		
		result.complete_discovery()
		return result
	
	async def _discover_openapi_endpoints(self, result: DiscoveryResult):
		"""Try to discover endpoints from OpenAPI/Swagger documentation"""
		common_docs_paths = ["/openapi.json", "/swagger.json", "/api-docs", "/docs", "/swagger/v1/swagger.json"]
		
		for path in common_docs_paths:
			try:
				doc_url = urljoin(self.base_url, path)
				response = await self.client.get(doc_url)
				
				if response.status_code == 200:
					spec = response.json()
					
					if "paths" in spec:  # OpenAPI spec
						for path, methods in spec["paths"].items():
							for method, details in methods.items():
								if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
									asset = await self._create_endpoint_asset(path, method.upper(), details)
									if should_include_asset(asset.name, self.config.include_patterns, self.config.exclude_patterns):
										result.add_asset(asset)
						return
			except Exception:
				continue
	
	async def _discover_common_endpoints(self, result: DiscoveryResult):
		"""Try to discover common REST endpoints"""
		common_paths = [
			"/api/v1/users", "/api/users", "/users",
			"/api/v1/products", "/api/products", "/products",
			"/api/v1/orders", "/api/orders", "/orders",
			"/api/v1/customers", "/api/customers", "/customers",
			"/api/health", "/health", "/status",
			"/api/v1", "/api"
		]
		
		for path in common_paths:
			try:
				url = urljoin(self.base_url, path)
				response = await self.client.get(url)
				
				if response.status_code in [200, 201]:
					asset = await self._create_endpoint_asset(path, "GET", {
						"summary": f"Discovered endpoint at {path}",
						"responses": {"200": {"description": "Success"}}
					})
					if should_include_asset(asset.name, self.config.include_patterns, self.config.exclude_patterns):
						result.add_asset(asset)
			except Exception:
				continue
	
	async def _discover_from_links(self, result: DiscoveryResult):
		"""Try to discover endpoints by following HATEOAS links"""
		try:
			response = await self.client.get(self.base_url)
			if response.status_code == 200:
				content_type = response.headers.get("content-type", "")
				if "application/json" in content_type:
					data = response.json()
					if isinstance(data, dict) and "_links" in data:
						for rel, link_info in data["_links"].items():
							if isinstance(link_info, dict) and "href" in link_info:
								path = link_info["href"]
								asset = await self._create_endpoint_asset(path, "GET", {
									"summary": f"HATEOAS link: {rel}"
								})
								if should_include_asset(asset.name, self.config.include_patterns, self.config.exclude_patterns):
									result.add_asset(asset)
		except Exception:
			pass
	
	async def _create_endpoint_asset(self, path: str, method: str, details: Dict[str, Any]) -> AssetMetadata:
		"""Create asset metadata for an API endpoint"""
		asset_name = f"{method} {path}"
		description = details.get("summary") or details.get("description") or f"{method} endpoint at {path}"
		
		asset = AssetMetadata(
			name=asset_name,
			asset_type="api_endpoint",
			source_system=self.source_system,
			full_name=f"{self.base_url}{path}",
			description=description,
			location=f"{self.base_url}{path}",
			properties={
				"method": method,
				"path": path,
				"base_url": self.base_url,
				"content_type": details.get("produces", ["application/json"]),
				"parameters": details.get("parameters", []),
				"responses": details.get("responses", {})
			}
		)
		
		# Extract schema information from parameters and responses
		columns = []
		
		# Add parameters as columns
		for param in details.get("parameters", []):
			if isinstance(param, dict):
				param_name = param.get("name", "unknown")
				param_type = param.get("type", "string")
				data_type = self._map_openapi_type(param_type)
				
				column = ColumnMetadata(
					name=param_name,
					data_type=data_type,
					is_nullable=not param.get("required", False),
					description=param.get("description"),
					classification_hints=["parameter"]
				)
				columns.append(column)
		
		# Add response schema as columns if available
		responses = details.get("responses", {})
		for status_code, response_info in responses.items():
			if isinstance(response_info, dict) and "schema" in response_info:
				schema = response_info["schema"]
				if isinstance(schema, dict) and "properties" in schema:
					for prop_name, prop_details in schema["properties"].items():
						prop_type = prop_details.get("type", "string")
						data_type = self._map_openapi_type(prop_type)
						
						column = ColumnMetadata(
							name=prop_name,
							data_type=data_type,
							description=prop_details.get("description"),
							classification_hints=["response_field"]
						)
						columns.append(column)
		
		asset.columns = columns
		asset.column_count = len(columns)
		asset.estimated_quality_score = self._estimate_quality_score(asset)
		
		return asset
	
	def _map_openapi_type(self, openapi_type: str) -> DataType:
		"""Map OpenAPI type to DataType enum"""
		type_mapping = {
			"string": DataType.STRING,
			"integer": DataType.INTEGER,
			"number": DataType.FLOAT,
			"boolean": DataType.BOOLEAN,
			"array": DataType.ARRAY,
			"object": DataType.OBJECT
		}
		return type_mapping.get(openapi_type, DataType.STRING)
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get detailed schema information for a specific endpoint"""
		if not self.is_connected:
			if not await self.connect():
				return None
		
		try:
			# Extract method and path from asset name
			parts = asset_name.split(" ", 1)
			if len(parts) != 2:
				return None
			
			method, path = parts
			url = urljoin(self.base_url, path)
			
			# Make a request to analyze the actual response structure
			if method == "GET":
				response = await self.client.get(url)
			else:
				return None  # For now, only support GET requests for schema discovery
			
			if response.status_code == 200:
				content_type = response.headers.get("content-type", "")
				if "application/json" in content_type:
					data = response.json()
					columns = await self._analyze_json_structure(data)
					
					asset = AssetMetadata(
						name=asset_name,
						asset_type="api_endpoint",
						source_system=self.source_system,
						full_name=url,
						description=f"API endpoint {method} {path}",
						location=url,
						columns=columns,
						column_count=len(columns)
					)
					
					asset.estimated_quality_score = self._estimate_quality_score(asset)
					return asset
			
		except Exception as e:
			await self._log_error(f"Error getting schema for {asset_name}: {str(e)}")
		
		return None
	
	async def _analyze_json_structure(self, data: Any, prefix: str = "") -> List[ColumnMetadata]:
		"""Analyze JSON structure and extract column metadata"""
		columns = []
		
		if isinstance(data, dict):
			for key, value in data.items():
				field_name = f"{prefix}.{key}" if prefix else key
				
				if isinstance(value, (dict, list)):
					# Nested structure
					data_type = DataType.OBJECT if isinstance(value, dict) else DataType.ARRAY
					column = ColumnMetadata(
						name=field_name,
						data_type=data_type,
						is_nullable=True
					)
					columns.append(column)
					
					# Recursively analyze nested structure
					if isinstance(value, dict) and len(columns) < 50:  # Limit depth
						nested_columns = await self._analyze_json_structure(value, field_name)
						columns.extend(nested_columns)
				else:
					# Simple value
					data_type = self._infer_data_type([value])
					column = ColumnMetadata(
						name=field_name,
						data_type=data_type,
						is_nullable=True,
						sample_values=[str(value)] if value is not None else []
					)
					columns.append(column)
		
		elif isinstance(data, list) and data:
			# Analyze first item in array
			nested_columns = await self._analyze_json_structure(data[0], prefix)
			columns.extend(nested_columns)
		
		return columns
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Sample data from an API endpoint"""
		if not self.is_connected:
			if not await self.connect():
				return []
		
		try:
			# Extract method and path from asset name
			parts = asset_name.split(" ", 1)
			if len(parts) != 2:
				return []
			
			method, path = parts
			
			# Only support GET for data sampling
			if method != "GET":
				return []
			
			url = urljoin(self.base_url, path)
			
			# Add limit parameter if the API supports it
			params = {}
			if "?" not in url:
				params["limit"] = min(limit, 100)
			
			response = await self.client.get(url, params=params)
			
			if response.status_code == 200:
				content_type = response.headers.get("content-type", "")
				if "application/json" in content_type:
					data = response.json()
					
					# Handle different response formats
					if isinstance(data, list):
						return data[:limit]
					elif isinstance(data, dict):
						# Check for common pagination patterns
						for key in ["data", "items", "results", "records"]:
							if key in data and isinstance(data[key], list):
								return data[key][:limit]
						
						# Return the object as a single-item list
						return [data]
					
		except Exception as e:
			await self._log_error(f"Error sampling data from {asset_name}: {str(e)}")
		
		return []


class GraphQLConnector(BaseConnector):
	"""GraphQL API metadata discovery connector"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.API
		self.source_system = "graphql"
		self.client = None
		self.endpoint_url = config.connection_string
		self.schema_cache = None
		self.auth_headers = {}
	
	async def connect(self) -> bool:
		"""Establish connection to GraphQL endpoint"""
		try:
			if httpx is None:
				raise ImportError("httpx library is required for GraphQL connector")
			
			# Setup authentication headers
			if self.config.username and self.config.password:
				import base64
				creds = base64.b64encode(f"{self.config.username}:{self.config.password}".encode()).decode()
				self.auth_headers["Authorization"] = f"Basic {creds}"
			
			# Add Bearer token if provided
			if "token" in self.config.additional_params:
				self.auth_headers["Authorization"] = f"Bearer {self.config.additional_params['token']}"
			
			# Add any additional headers
			self.auth_headers.update(self.config.additional_params.get("headers", {}))
			self.auth_headers["Content-Type"] = "application/json"
			
			# Create HTTP client
			self.client = httpx.AsyncClient(
				headers=self.auth_headers,
				timeout=self.config.connection_timeout,
				verify=self.config.use_ssl
			)
			
			self.is_connected = True
			return True
		except Exception as e:
			await self._log_error(f"Failed to connect to GraphQL endpoint: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Close connection to GraphQL endpoint"""
		if self.client:
			await self.client.aclose()
			self.client = None
		self.is_connected = False
		self.schema_cache = None
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test connection to GraphQL endpoint"""
		if not self.is_connected:
			if not await self.connect():
				return {"status": "error", "message": "Failed to connect to GraphQL endpoint"}
		
		try:
			# Test with introspection query
			introspection_query = {
				"query": "{ __schema { queryType { name } } }"
			}
			
			response = await self.client.post(self.endpoint_url, json=introspection_query)
			
			if response.status_code == 200:
				result = response.json()
				if "data" in result and result["data"] is not None:
					return {
						"status": "success",
						"message": "Connected to GraphQL endpoint successfully",
						"schema_available": True
					}
				else:
					return {
						"status": "warning",
						"message": "Connected but introspection may be disabled",
						"errors": result.get("errors", [])
					}
			else:
				return {
					"status": "error",
					"message": f"HTTP {response.status_code}: {response.text}"
				}
			
		except Exception as e:
			return {"status": "error", "message": f"Connection test failed: {str(e)}"}
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover GraphQL schema and extract types as assets"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		
		if not self.is_connected:
			if not await self.connect():
				result.add_error("Failed to connect to GraphQL endpoint")
				result.complete_discovery()
				return result
		
		try:
			# Get full schema introspection
			schema = await self._get_schema_introspection()
			
			if not schema:
				result.add_error("Failed to retrieve GraphQL schema")
				result.complete_discovery()
				return result
			
			self.schema_cache = schema
			
			# Extract types as assets
			if "types" in schema:
				for type_def in schema["types"]:
					if self._should_include_type(type_def):
						asset = await self._create_type_asset(type_def)
						if asset and should_include_asset(asset.name, self.config.include_patterns, self.config.exclude_patterns):
							result.add_asset(asset)
			
			# Extract queries, mutations, and subscriptions
			for operation_type in ["queryType", "mutationType", "subscriptionType"]:
				if operation_type in schema and schema[operation_type]:
					type_name = schema[operation_type]["name"]
					type_def = next((t for t in schema["types"] if t["name"] == type_name), None)
					if type_def and "fields" in type_def:
						for field in type_def["fields"]:
							asset = await self._create_operation_asset(field, operation_type)
							if asset and should_include_asset(asset.name, self.config.include_patterns, self.config.exclude_patterns):
								result.add_asset(asset)
			
		except Exception as e:
			result.add_error(f"Error during GraphQL schema discovery: {str(e)}")
		
		result.complete_discovery()
		return result
	
	async def _get_schema_introspection(self) -> Optional[Dict[str, Any]]:
		"""Get full GraphQL schema through introspection"""
		introspection_query = {
			"query": """
			query IntrospectionQuery {
				__schema {
					queryType { name }
					mutationType { name }
					subscriptionType { name }
					types {
						...FullType
					}
					directives {
						name
						description
						location
						args {
							...InputValue
						}
					}
				}
			}

			fragment FullType on __Type {
				kind
				name
				description
				fields(includeDeprecated: true) {
					name
					description
					args {
						...InputValue
					}
					type {
						...TypeRef
					}
					isDeprecated
					deprecationReason
				}
				inputFields {
					...InputValue
				}
				interfaces {
					...TypeRef
				}
				enumValues(includeDeprecated: true) {
					name
					description
					isDeprecated
					deprecationReason
				}
				possibleTypes {
					...TypeRef
				}
			}

			fragment InputValue on __InputValue {
				name
				description
				type { ...TypeRef }
				defaultValue
			}

			fragment TypeRef on __Type {
				kind
				name
				ofType {
					kind
					name
					ofType {
						kind
						name
						ofType {
							kind
							name
							ofType {
								kind
								name
								ofType {
									kind
									name
									ofType {
										kind
										name
										ofType {
											kind
											name
										}
									}
								}
							}
						}
					}
				}
			}
			"""
		}
		
		try:
			response = await self.client.post(self.endpoint_url, json=introspection_query)
			
			if response.status_code == 200:
				result = response.json()
				if "data" in result and "__schema" in result["data"]:
					return result["data"]["__schema"]
			
		except Exception as e:
			await self._log_error(f"Error getting GraphQL schema: {str(e)}")
		
		return None
	
	def _should_include_type(self, type_def: Dict[str, Any]) -> bool:
		"""Determine if a GraphQL type should be included as an asset"""
		if not type_def or "name" not in type_def:
			return False
		
		name = type_def["name"]
		kind = type_def.get("kind", "")
		
		# Skip built-in scalar types and introspection types
		if name.startswith("__") or kind in ["SCALAR"] and name in ["String", "Int", "Float", "Boolean", "ID"]:
			return False
		
		# Include OBJECT, INPUT_OBJECT, ENUM, INTERFACE, UNION types
		return kind in ["OBJECT", "INPUT_OBJECT", "ENUM", "INTERFACE", "UNION"]
	
	async def _create_type_asset(self, type_def: Dict[str, Any]) -> Optional[AssetMetadata]:
		"""Create asset metadata for a GraphQL type"""
		name = type_def.get("name")
		kind = type_def.get("kind")
		description = type_def.get("description") or f"GraphQL {kind.lower()}: {name}"
		
		asset = AssetMetadata(
			name=name,
			asset_type=f"graphql_{kind.lower()}",
			source_system=self.source_system,
			description=description,
			location=self.endpoint_url,
			properties={
				"kind": kind,
				"endpoint": self.endpoint_url
			}
		)
		
		# Extract fields as columns
		columns = []
		
		# Handle different type kinds
		if kind == "OBJECT" and "fields" in type_def:
			for field in type_def["fields"]:
				column = await self._create_field_column(field)
				if column:
					columns.append(column)
		
		elif kind == "INPUT_OBJECT" and "inputFields" in type_def:
			for field in type_def["inputFields"]:
				column = await self._create_input_field_column(field)
				if column:
					columns.append(column)
		
		elif kind == "ENUM" and "enumValues" in type_def:
			enum_values = [value["name"] for value in type_def["enumValues"] if not value.get("isDeprecated")]
			column = ColumnMetadata(
				name="value",
				data_type=DataType.STRING,
				description="Enum value",
				sample_values=enum_values[:10]
			)
			columns.append(column)
		
		asset.columns = columns
		asset.column_count = len(columns)
		asset.estimated_quality_score = self._estimate_quality_score(asset)
		
		return asset
	
	async def _create_operation_asset(self, field: Dict[str, Any], operation_type: str) -> Optional[AssetMetadata]:
		"""Create asset metadata for a GraphQL operation (query/mutation/subscription)"""
		field_name = field.get("name")
		description = field.get("description") or f"GraphQL {operation_type}: {field_name}"
		
		opertion_name = operation_type.replace("Type", "")
		asset_name = f"{opertion_name}_{field_name}"
		
		asset = AssetMetadata(
			name=asset_name,
			asset_type=f"graphql_{opertion_name}",
			source_system=self.source_system,
			description=description,
			location=self.endpoint_url,
			properties={
				"operation_type": opertion_name,
				"field_name": field_name,
				"endpoint": self.endpoint_url
			}
		)
		
		# Extract arguments as columns
		columns = []
		
		# Add arguments
		for arg in field.get("args", []):
			column = await self._create_argument_column(arg)
			if column:
				columns.append(column)
		
		# Add return type information
		return_type = field.get("type")
		if return_type:
			return_type_name = self._extract_type_name(return_type)
			column = ColumnMetadata(
				name="__return_type",
				data_type=DataType.OBJECT,
				description=f"Returns: {return_type_name}",
				classification_hints=["return_type"]
			)
			columns.append(column)
		
		asset.columns = columns
		asset.column_count = len(columns)
		asset.estimated_quality_score = self._estimate_quality_score(asset)
		
		return asset
	
	async def _create_field_column(self, field: Dict[str, Any]) -> Optional[ColumnMetadata]:
		"""Create column metadata for a GraphQL field"""
		name = field.get("name")
		description = field.get("description")
		field_type = field.get("type")
		
		data_type = self._map_graphql_type(field_type)
		type_name = self._extract_type_name(field_type)
		is_nullable = not self._is_non_null_type(field_type)
		
		column = ColumnMetadata(
			name=name,
			data_type=data_type,
			is_nullable=is_nullable,
			description=description,
			classification_hints=["field"],
			properties={"graphql_type": type_name}
		)
		
		return column
	
	async def _create_input_field_column(self, field: Dict[str, Any]) -> Optional[ColumnMetadata]:
		"""Create column metadata for a GraphQL input field"""
		name = field.get("name")
		description = field.get("description")
		field_type = field.get("type")
		default_value = field.get("defaultValue")
		
		data_type = self._map_graphql_type(field_type)
		type_name = self._extract_type_name(field_type)
		is_nullable = not self._is_non_null_type(field_type)
		
		column = ColumnMetadata(
			name=name,
			data_type=data_type,
			is_nullable=is_nullable,
			description=description,
			default_value=default_value,
			classification_hints=["input_field"],
			properties={"graphql_type": type_name}
		)
		
		return column
	
	async def _create_argument_column(self, arg: Dict[str, Any]) -> Optional[ColumnMetadata]:
		"""Create column metadata for a GraphQL argument"""
		name = arg.get("name")
		description = arg.get("description")
		arg_type = arg.get("type")
		default_value = arg.get("defaultValue")
		
		data_type = self._map_graphql_type(arg_type)
		type_name = self._extract_type_name(arg_type)
		is_nullable = not self._is_non_null_type(arg_type)
		
		column = ColumnMetadata(
			name=name,
			data_type=data_type,
			is_nullable=is_nullable,
			description=description,
			default_value=default_value,
			classification_hints=["argument"],
			properties={"graphql_type": type_name}
		)
		
		return column
	
	def _map_graphql_type(self, graphql_type: Dict[str, Any]) -> DataType:
		"""Map GraphQL type to DataType enum"""
		if not graphql_type:
			return DataType.UNKNOWN
		
		# Handle wrapping types (NON_NULL, LIST)
		kind = graphql_type.get("kind")
		if kind == "NON_NULL" or kind == "LIST":
			of_type = graphql_type.get("ofType")
			if of_type:
				if kind == "LIST":
					return DataType.ARRAY
				return self._map_graphql_type(of_type)
		
		# Handle named types
		type_name = graphql_type.get("name", "")
		
		# Map GraphQL scalar types
		scalar_mapping = {
			"String": DataType.STRING,
			"Int": DataType.INTEGER,
			"Float": DataType.FLOAT,
			"Boolean": DataType.BOOLEAN,
			"ID": DataType.STRING,
			"Date": DataType.DATE,
			"DateTime": DataType.DATETIME,
			"Time": DataType.TIMESTAMP
		}
		
		if type_name in scalar_mapping:
			return scalar_mapping[type_name]
		
		# Handle complex types
		if kind == "OBJECT":
			return DataType.OBJECT
		elif kind == "ENUM":
			return DataType.STRING
		elif kind == "SCALAR":
			return DataType.STRING  # Custom scalar
		
		return DataType.UNKNOWN
	
	def _extract_type_name(self, graphql_type: Dict[str, Any]) -> str:
		"""Extract the base type name from a GraphQL type"""
		if not graphql_type:
			return "Unknown"
		
		kind = graphql_type.get("kind")
		name = graphql_type.get("name")
		
		if name:
			return name
		
		# Handle wrapping types
		if kind == "NON_NULL":
			of_type = graphql_type.get("ofType")
			if of_type:
				return f"{self._extract_type_name(of_type)}!"
		
		if kind == "LIST":
			of_type = graphql_type.get("ofType")
			if of_type:
				return f"[{self._extract_type_name(of_type)}]"
		
		return kind or "Unknown"
	
	def _is_non_null_type(self, graphql_type: Dict[str, Any]) -> bool:
		"""Check if GraphQL type is non-null"""
		return graphql_type and graphql_type.get("kind") == "NON_NULL"
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get detailed schema information for a specific GraphQL asset"""
		if not self.schema_cache:
			# Try to get schema if not cached
			if not self.is_connected:
				if not await self.connect():
					return None
			
			self.schema_cache = await self._get_schema_introspection()
			
		if not self.schema_cache:
			return None
		
		try:
			# Find the type in the schema
			for type_def in self.schema_cache.get("types", []):
				if type_def.get("name") == asset_name:
					return await self._create_type_asset(type_def)
			
			# Check if it's an operation
			for operation_type in ["queryType", "mutationType", "subscriptionType"]:
				if operation_type in self.schema_cache and self.schema_cache[operation_type]:
					type_name = self.schema_cache[operation_type]["name"]
					type_def = next((t for t in self.schema_cache["types"] if t["name"] == type_name), None)
					if type_def and "fields" in type_def:
						for field in type_def["fields"]:
							opertion_name = operation_type.replace("Type", "")
							field_asset_name = f"{opertion_name}_{field['name']}"
							if field_asset_name == asset_name:
								return await self._create_operation_asset(field, operation_type)
			
		except Exception as e:
			await self._log_error(f"Error getting schema for {asset_name}: {str(e)}")
		
		return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Sample data from a GraphQL query"""
		if not self.is_connected:
			if not await self.connect():
				return []
		
		try:
			# Check if this is a query operation
			if asset_name.startswith("query_"):
				query_name = asset_name.replace("query_", "")
				
				# Create a basic query
				query = {
					"query": f"{{ {query_name} }}"
				}
				
				response = await self.client.post(self.endpoint_url, json=query)
				
				if response.status_code == 200:
					result = response.json()
					if "data" in result and result["data"] is not None:
						data = result["data"].get(query_name)
						
						# Handle different response formats
						if isinstance(data, list):
							return data[:limit]
						elif isinstance(data, dict):
							return [data]
						elif data is not None:
							return [{query_name: data}]
			
			# For type assets, return sample schema information
			if self.schema_cache:
				for type_def in self.schema_cache.get("types", []):
					if type_def.get("name") == asset_name:
						# Return field information as sample data
						fields = type_def.get("fields", [])
						return [{
							"field_name": field["name"],
							"field_type": self._extract_type_name(field.get("type", {})),
							"description": field.get("description")
						} for field in fields[:limit]]
			
		except Exception as e:
			await self._log_error(f"Error sampling data from {asset_name}: {str(e)}")
		
		return []


class KafkaConnector(BaseConnector):
	"""Apache Kafka metadata discovery connector"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.STREAMING
		self.source_system = "kafka"
		self.admin_client = None
		self.consumer = None
		self.bootstrap_servers = config.connection_string.split(",")
		self.topics_metadata = {}
	
	async def connect(self) -> bool:
		"""Establish connection to Kafka cluster"""
		try:
			if KafkaAdminClient is None:
				raise ImportError("kafka-python library is required for Kafka connector")
			
			# Setup connection config
			config = {
				'bootstrap_servers': self.bootstrap_servers,
				'client_id': 'apg_metadata_connector',
				'request_timeout_ms': self.config.connection_timeout * 1000,
				'api_version': (0, 10, 1)
			}
			
			# Add authentication if provided
			if self.config.username and self.config.password:
				config.update({
					'security_protocol': 'SASL_PLAINTEXT',
					'sasl_mechanism': 'PLAIN',
					'sasl_plain_username': self.config.username,
					'sasl_plain_password': self.config.password
				})
			
			# Add SSL if enabled
			if self.config.use_ssl:
				config['security_protocol'] = 'SSL'
				if self.config.ssl_cert_path:
					config['ssl_certfile'] = self.config.ssl_cert_path
				if self.config.ssl_key_path:
					config['ssl_keyfile'] = self.config.ssl_key_path
			
			# Add additional parameters
			config.update(self.config.additional_params)
			
			# Create admin client
			self.admin_client = KafkaAdminClient(**config)
			
			# Test the connection
			metadata = self.admin_client.describe_cluster()
			if not metadata:
				raise Exception("Failed to get cluster metadata")
			
			self.is_connected = True
			return True
			
		except Exception as e:
			await self._log_error(f"Failed to connect to Kafka cluster: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Close connection to Kafka cluster"""
		if self.admin_client:
			try:
				self.admin_client.close()
			except:
				pass
			self.admin_client = None
		
		if self.consumer:
			try:
				self.consumer.close()
			except:
				pass
			self.consumer = None
		
		self.is_connected = False
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test connection to Kafka cluster"""
		if not self.is_connected:
			if not await self.connect():
				return {"status": "error", "message": "Failed to connect to Kafka cluster"}
		
		try:
			# Get cluster metadata
			metadata = self.admin_client.describe_cluster()
			
			return {
				"status": "success",
				"message": "Connected to Kafka cluster successfully",
				"cluster_id": getattr(metadata, 'cluster_id', 'unknown'),
				"brokers": len(getattr(metadata, 'brokers', [])),
				"bootstrap_servers": self.bootstrap_servers
			}
			
		except Exception as e:
			return {"status": "error", "message": f"Connection test failed: {str(e)}"}
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover Kafka topics and their metadata"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		
		if not self.is_connected:
			if not await self.connect():
				result.add_error("Failed to connect to Kafka cluster")
				result.complete_discovery()
				return result
		
		try:
			# Get cluster metadata to discover topics
			metadata = self.admin_client.describe_cluster()
			topics = self.admin_client.list_topics()
			
			for topic_name in topics:
				try:
					# Skip internal topics if not explicitly included
					if topic_name.startswith('__') and not self.config.additional_params.get('include_internal_topics', False):
						continue
					
					if should_include_asset(topic_name, self.config.include_patterns, self.config.exclude_patterns):
						asset = await self._create_topic_asset(topic_name)
						if asset:
							result.add_asset(asset)
					
				except Exception as e:
					result.add_warning(f"Error processing topic {topic_name}: {str(e)}")
					continue
			
		except Exception as e:
			result.add_error(f"Error during Kafka topic discovery: {str(e)}")
		
		result.complete_discovery()
		return result
	
	async def _create_topic_asset(self, topic_name: str) -> Optional[AssetMetadata]:
		"""Create asset metadata for a Kafka topic"""
		try:
			# Get topic details
			topic_details = self.admin_client.describe_topics([topic_name])
			topic_info = topic_details.get(topic_name)
			
			if not topic_info:
				return None
			
			# Get partition information
			partitions = getattr(topic_info, 'partitions', {})
			partition_count = len(partitions)
			
			# Get topic configuration
			configs = {}
			try:
				config_result = self.admin_client.describe_configs(
					config_resources=[('TOPIC', topic_name)]
				)
				if topic_name in config_result:
					configs = config_result[topic_name]
			except:
				pass
			
			asset = AssetMetadata(
				name=topic_name,
				asset_type="kafka_topic",
				source_system=self.source_system,
				description=f"Kafka topic: {topic_name}",
				location=f"kafka://{','.join(self.bootstrap_servers)}/{topic_name}",
				properties={
					"partition_count": partition_count,
					"replication_factor": getattr(topic_info, 'replication_factor', 1),
					"configs": {k: str(v) for k, v in configs.items()} if isinstance(configs, dict) else {},
					"bootstrap_servers": self.bootstrap_servers
				}
			)
			
			# Create columns for message structure (if we can sample)
			columns = []
			
			# Always add basic message metadata columns
			columns.extend([
				ColumnMetadata(
					name="__key",
					data_type=DataType.STRING,
					description="Message key",
					is_nullable=True,
					classification_hints=["message_key"]
				),
				ColumnMetadata(
					name="__value",
					data_type=DataType.STRING,
					description="Message value",
					is_nullable=True,
					classification_hints=["message_value"]
				),
				ColumnMetadata(
					name="__partition",
					data_type=DataType.INTEGER,
					description="Message partition",
					is_nullable=False,
					classification_hints=["partition_id"]
				),
				ColumnMetadata(
					name="__offset",
					data_type=DataType.INTEGER,
					description="Message offset",
					is_nullable=False,
					classification_hints=["message_offset"]
				),
				ColumnMetadata(
					name="__timestamp",
					data_type=DataType.TIMESTAMP,
					description="Message timestamp",
					is_nullable=True,
					classification_hints=["timestamp"]
				)
			])
			
			# Try to sample messages to infer schema
			if self.config.enable_schema_inference:
				try:
					sample_messages = await self._sample_topic_messages(topic_name, 10)
					if sample_messages:
						inferred_columns = await self._infer_message_schema(sample_messages)
						columns.extend(inferred_columns)
				except Exception as e:
					await self._log_warning(f"Could not infer schema for topic {topic_name}: {str(e)}")
			
			asset.columns = columns
			asset.column_count = len(columns)
			asset.estimated_quality_score = self._estimate_quality_score(asset)
			
			# Cache topic metadata
			self.topics_metadata[topic_name] = asset
			
			return asset
			
		except Exception as e:
			await self._log_error(f"Error creating asset for topic {topic_name}: {str(e)}")
			return None
	
	async def _sample_topic_messages(self, topic_name: str, max_messages: int = 10) -> List[Dict[str, Any]]:
		"""Sample messages from a Kafka topic"""
		messages = []
		consumer = None
		
		try:
			# Create consumer for sampling
			consumer_config = {
				'bootstrap_servers': self.bootstrap_servers,
				'group_id': f'apg_metadata_sampler_{topic_name}',
				'auto_offset_reset': 'latest',  # Start from latest to avoid consuming too much
				'enable_auto_commit': False,
				'consumer_timeout_ms': 5000,  # 5 second timeout
				'max_poll_records': max_messages,
				'value_deserializer': lambda x: x.decode('utf-8', errors='ignore') if x else None,
				'key_deserializer': lambda x: x.decode('utf-8', errors='ignore') if x else None
			}
			
			# Add authentication if available
			if self.config.username and self.config.password:
				consumer_config.update({
					'security_protocol': 'SASL_PLAINTEXT',
					'sasl_mechanism': 'PLAIN',
					'sasl_plain_username': self.config.username,
					'sasl_plain_password': self.config.password
				})
			
			consumer = KafkaConsumer(topic_name, **consumer_config)
			
			# Poll for messages
			message_count = 0
			for message in consumer:
				if message_count >= max_messages:
					break
				
				messages.append({
					"key": message.key,
					"value": message.value,
					"partition": message.partition,
					"offset": message.offset,
					"timestamp": message.timestamp
				})
				message_count += 1
			
		except Exception as e:
			await self._log_warning(f"Error sampling messages from {topic_name}: {str(e)}")
			
		finally:
			if consumer:
				try:
					consumer.close()
				except:
					pass
		
		return messages
	
	async def _infer_message_schema(self, messages: List[Dict[str, Any]]) -> List[ColumnMetadata]:
		"""Infer schema from sample messages"""
		columns = []
		value_schemas = []
		
		# Analyze message values
		for message in messages:
			value = message.get("value")
			if value:
				try:
					# Try to parse as JSON
					if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
						parsed_value = json.loads(value)
						value_schemas.append(parsed_value)
				except (json.JSONDecodeError, ValueError):
					# Not JSON, treat as string
					value_schemas.append(value)
		
		# If we have JSON schemas, analyze the structure
		if value_schemas:
			for schema in value_schemas:
				if isinstance(schema, dict):
					for field_name, field_value in schema.items():
						data_type = self._infer_data_type([field_value])
						
						# Check if column already exists
						existing_column = next((c for c in columns if c.name == field_name), None)
						if not existing_column:
							column = ColumnMetadata(
								name=field_name,
								data_type=data_type,
								is_nullable=True,
								description=f"Field from message value: {field_name}",
								sample_values=[str(field_value)] if field_value is not None else [],
								classification_hints=["message_field"]
							)
							columns.append(column)
						else:
							# Update existing column with more sample values
							if str(field_value) not in existing_column.sample_values:
								existing_column.sample_values.append(str(field_value))
								existing_column.sample_values = existing_column.sample_values[:10]  # Limit samples
		
		return columns
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get detailed schema information for a Kafka topic"""
		if asset_name in self.topics_metadata:
			return self.topics_metadata[asset_name]
		
		if not self.is_connected:
			if not await self.connect():
				return None
		
		try:
			# Create asset for the specific topic
			asset = await self._create_topic_asset(asset_name)
			return asset
			
		except Exception as e:
			await self._log_error(f"Error getting schema for topic {asset_name}: {str(e)}")
		
		return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Sample data from a Kafka topic"""
		if not self.is_connected:
			if not await self.connect():
				return []
		
		try:
			# Sample messages from the topic
			messages = await self._sample_topic_messages(asset_name, min(limit, 100))
			
			# Format messages for return
			formatted_messages = []
			for message in messages:
				value = message.get("value")
				
				# Try to parse JSON value
				parsed_value = value
				if isinstance(value, str):
					try:
						parsed_value = json.loads(value)
					except (json.JSONDecodeError, ValueError):
						pass
				
				formatted_message = {
					"__key": message.get("key"),
					"__value": parsed_value,
					"__partition": message.get("partition"),
					"__offset": message.get("offset"),
					"__timestamp": message.get("timestamp")
				}
				
				# If value is a dict, merge its fields
				if isinstance(parsed_value, dict):
					formatted_message.update(parsed_value)
				
				formatted_messages.append(formatted_message)
			
			return formatted_messages
			
		except Exception as e:
			await self._log_error(f"Error sampling data from topic {asset_name}: {str(e)}")
		
		return []
	
	async def _log_warning(self, message: str):
		"""Log warning message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] {self.__class__.__name__} WARNING: {message}")