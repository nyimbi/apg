"""
APG Workflow & Business Process Management - Integration Hub

Enterprise integration hub for connecting workflows with external systems,
APIs, databases, and services with intelligent mapping and transformation.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
import httpx
import ssl
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict
import base64
import hashlib
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse

from models import (
	APGTenantContext, WBPMServiceResponse, WBPMPagedResponse
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Integration Core Classes
# =============================================================================

class ConnectorType(str, Enum):
	"""Types of system connectors."""
	REST_API = "rest_api"
	SOAP_WS = "soap_ws"
	DATABASE = "database"
	FILE_SYSTEM = "file_system"
	EMAIL = "email"
	FTP_SFTP = "ftp_sftp"
	MESSAGE_QUEUE = "message_queue"
	WEBHOOK = "webhook"
	CUSTOM = "custom"


class AuthenticationType(str, Enum):
	"""Authentication types for integrations."""
	NONE = "none"
	BASIC = "basic"
	BEARER_TOKEN = "bearer_token"
	API_KEY = "api_key"
	OAUTH2 = "oauth2"
	CERTIFICATE = "certificate"
	CUSTOM_HEADER = "custom_header"


class DataFormat(str, Enum):
	"""Data exchange formats."""
	JSON = "json"
	XML = "xml"
	CSV = "csv"
	PLAIN_TEXT = "plain_text"
	BINARY = "binary"
	FORM_DATA = "form_data"


class MappingType(str, Enum):
	"""Data mapping types."""
	DIRECT = "direct"
	EXPRESSION = "expression"
	LOOKUP = "lookup"
	TRANSFORMATION = "transformation"
	CONDITIONAL = "conditional"


@dataclass
class ConnectionCredentials:
	"""Connection credentials for external systems."""
	credential_id: str = field(default_factory=lambda: f"cred_{uuid.uuid4().hex}")
	tenant_id: str = ""
	auth_type: AuthenticationType = AuthenticationType.NONE
	username: Optional[str] = None
	password: Optional[str] = None  # Encrypted in production
	api_key: Optional[str] = None
	bearer_token: Optional[str] = None
	oauth2_config: Dict[str, Any] = field(default_factory=dict)
	certificate_data: Optional[str] = None
	custom_headers: Dict[str, str] = field(default_factory=dict)
	expires_at: Optional[datetime] = None
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)
	
	def is_expired(self) -> bool:
		"""Check if credentials are expired."""
		return self.expires_at and datetime.utcnow() > self.expires_at


@dataclass
class SystemConnector:
	"""External system connector configuration."""
	connector_id: str = field(default_factory=lambda: f"conn_{uuid.uuid4().hex}")
	tenant_id: str = ""
	connector_name: str = ""
	connector_type: ConnectorType = ConnectorType.REST_API
	base_url: Optional[str] = None
	connection_config: Dict[str, Any] = field(default_factory=dict)
	credential_id: str = ""
	timeout_seconds: int = 30
	retry_config: Dict[str, Any] = field(default_factory=dict)
	rate_limit_config: Dict[str, Any] = field(default_factory=dict)
	health_check_config: Dict[str, Any] = field(default_factory=dict)
	tags: List[str] = field(default_factory=list)
	is_active: bool = True
	created_by: str = ""
	updated_by: str = ""
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)
	last_health_check: Optional[datetime] = None
	health_status: str = "unknown"  # healthy, unhealthy, unknown


@dataclass
class DataMapping:
	"""Data field mapping configuration."""
	mapping_id: str = field(default_factory=lambda: f"map_{uuid.uuid4().hex}")
	source_field: str = ""
	target_field: str = ""
	mapping_type: MappingType = MappingType.DIRECT
	transformation_expression: Optional[str] = None
	lookup_table: Dict[str, Any] = field(default_factory=dict)
	default_value: Any = None
	is_required: bool = False
	validation_rules: List[str] = field(default_factory=list)


@dataclass
class IntegrationEndpoint:
	"""Integration endpoint configuration."""
	endpoint_id: str = field(default_factory=lambda: f"endpoint_{uuid.uuid4().hex}")
	tenant_id: str = ""
	endpoint_name: str = ""
	connector_id: str = ""
	endpoint_path: str = ""
	http_method: str = "GET"
	input_format: DataFormat = DataFormat.JSON
	output_format: DataFormat = DataFormat.JSON
	input_mappings: List[DataMapping] = field(default_factory=list)
	output_mappings: List[DataMapping] = field(default_factory=list)
	request_template: Optional[str] = None
	response_schema: Dict[str, Any] = field(default_factory=dict)
	error_handling: Dict[str, Any] = field(default_factory=dict)
	cache_config: Dict[str, Any] = field(default_factory=dict)
	tags: List[str] = field(default_factory=list)
	is_active: bool = True
	created_by: str = ""
	updated_by: str = ""
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IntegrationExecution:
	"""Integration execution record."""
	execution_id: str = field(default_factory=lambda: f"exec_{uuid.uuid4().hex}")
	tenant_id: str = ""
	endpoint_id: str = ""
	process_instance_id: Optional[str] = None
	task_id: Optional[str] = None
	input_data: Dict[str, Any] = field(default_factory=dict)
	output_data: Dict[str, Any] = field(default_factory=dict)
	execution_status: str = "pending"  # pending, success, failed, timeout
	error_message: Optional[str] = None
	response_code: Optional[int] = None
	execution_time_ms: float = 0.0
	retry_count: int = 0
	started_at: datetime = field(default_factory=datetime.utcnow)
	completed_at: Optional[datetime] = None


# =============================================================================
# Data Transformation Engine
# =============================================================================

class DataTransformationEngine:
	"""Transform data between different formats and structures."""
	
	def __init__(self):
		self.transformation_functions = self._initialize_transformation_functions()
	
	async def apply_mappings(
		self,
		source_data: Dict[str, Any],
		mappings: List[DataMapping],
		context: Dict[str, Any] = None
	) -> Dict[str, Any]:
		"""Apply data mappings to transform source data."""
		target_data = {}
		context = context or {}
		
		for mapping in mappings:
			try:
				source_value = self._get_source_value(source_data, mapping.source_field)
				target_value = await self._transform_value(source_value, mapping, context)
				
				if target_value is not None or mapping.is_required:
					self._set_target_value(target_data, mapping.target_field, target_value)
				
			except Exception as e:
				logger.error(f"Error applying mapping {mapping.mapping_id}: {e}")
				if mapping.is_required:
					raise
		
		return target_data
	
	async def _transform_value(
		self,
		source_value: Any,
		mapping: DataMapping,
		context: Dict[str, Any]
	) -> Any:
		"""Transform single value based on mapping type."""
		if source_value is None and mapping.default_value is not None:
			source_value = mapping.default_value
		
		if mapping.mapping_type == MappingType.DIRECT:
			return source_value
		
		elif mapping.mapping_type == MappingType.EXPRESSION:
			return await self._evaluate_expression(
				mapping.transformation_expression, source_value, context
			)
		
		elif mapping.mapping_type == MappingType.LOOKUP:
			return mapping.lookup_table.get(str(source_value), source_value)
		
		elif mapping.mapping_type == MappingType.TRANSFORMATION:
			return await self._apply_transformation_function(
				mapping.transformation_expression, source_value, context
			)
		
		elif mapping.mapping_type == MappingType.CONDITIONAL:
			return await self._apply_conditional_mapping(
				mapping.transformation_expression, source_value, context
			)
		
		return source_value
	
	def _get_source_value(self, data: Dict[str, Any], field_path: str) -> Any:
		"""Get value from nested data structure using dot notation."""
		parts = field_path.split('.')
		current = data
		
		for part in parts:
			if isinstance(current, dict) and part in current:
				current = current[part]
			elif isinstance(current, list) and part.isdigit():
				index = int(part)
				current = current[index] if 0 <= index < len(current) else None
			else:
				return None
		
		return current
	
	def _set_target_value(self, data: Dict[str, Any], field_path: str, value: Any) -> None:
		"""Set value in nested data structure using dot notation."""
		parts = field_path.split('.')
		current = data
		
		for part in parts[:-1]:
			if part not in current:
				current[part] = {}
			current = current[part]
		
		current[parts[-1]] = value
	
	async def _evaluate_expression(
		self,
		expression: str,
		source_value: Any,
		context: Dict[str, Any]
	) -> Any:
		"""Evaluate transformation expression."""
		if not expression:
			return source_value
		
		try:
			# Create safe evaluation context
			eval_context = {
				'value': source_value,
				'context': context,
				**self.transformation_functions
			}
			
			# Security: Only allow specific built-ins
			safe_builtins = {
				'str': str,
				'int': int,
				'float': float,
				'bool': bool,
				'len': len,
				'abs': abs,
				'min': min,
				'max': max,
				'round': round
			}
			
			# Evaluate expression in restricted environment
			return eval(expression, {"__builtins__": safe_builtins}, eval_context)
			
		except Exception as e:
			logger.error(f"Expression evaluation error: {e}")
			return source_value
	
	async def _apply_transformation_function(
		self,
		function_name: str,
		source_value: Any,
		context: Dict[str, Any]
	) -> Any:
		"""Apply named transformation function."""
		if function_name in self.transformation_functions:
			func = self.transformation_functions[function_name]
			return await func(source_value, context)
		
		return source_value
	
	async def _apply_conditional_mapping(
		self,
		condition_expr: str,
		source_value: Any,
		context: Dict[str, Any]
	) -> Any:
		"""Apply conditional mapping logic."""
		# Parse condition expression (simplified implementation)
		# Format: "condition ? true_value : false_value"
		if '?' in condition_expr and ':' in condition_expr:
			condition, values = condition_expr.split('?', 1)
			true_value, false_value = values.split(':', 1)
			
			# Evaluate condition
			condition_result = await self._evaluate_expression(
				condition.strip(), source_value, context
			)
			
			if condition_result:
				return await self._evaluate_expression(
					true_value.strip(), source_value, context
				)
			else:
				return await self._evaluate_expression(
					false_value.strip(), source_value, context
				)
		
		return source_value
	
	def _initialize_transformation_functions(self) -> Dict[str, Callable]:
		"""Initialize built-in transformation functions."""
		return {
			'upper': lambda v, c: str(v).upper() if v else v,
			'lower': lambda v, c: str(v).lower() if v else v,
			'trim': lambda v, c: str(v).strip() if v else v,
			'to_date': lambda v, c: self._parse_date(v),
			'format_date': lambda v, c: self._format_date(v, c.get('format', '%Y-%m-%d')),
			'to_number': lambda v, c: float(v) if v and str(v).replace('.', '').isdigit() else 0,
			'concat': lambda v, c: str(v) + str(c.get('suffix', '')) if v else v,
			'prefix': lambda v, c: str(c.get('prefix', '')) + str(v) if v else v,
			'default': lambda v, c: v if v is not None else c.get('default_value'),
			'substring': lambda v, c: str(v)[c.get('start', 0):c.get('end')] if v else v
		}
	
	def _parse_date(self, value: Any) -> Optional[datetime]:
		"""Parse date from various formats."""
		if not value:
			return None
		
		try:
			if isinstance(value, datetime):
				return value
			
			# Try common date formats
			formats = [
				'%Y-%m-%d',
				'%Y-%m-%d %H:%M:%S',
				'%d/%m/%Y',
				'%m/%d/%Y',
				'%Y-%m-%dT%H:%M:%S',
				'%Y-%m-%dT%H:%M:%SZ'
			]
			
			for fmt in formats:
				try:
					return datetime.strptime(str(value), fmt)
				except ValueError:
					continue
			
			return None
			
		except Exception:
			return None
	
	def _format_date(self, value: Any, format_str: str) -> Optional[str]:
		"""Format date to string."""
		if not value:
			return None
		
		try:
			if isinstance(value, str):
				value = self._parse_date(value)
			
			if isinstance(value, datetime):
				return value.strftime(format_str)
			
			return None
			
		except Exception:
			return None


# =============================================================================
# Connection Manager
# =============================================================================

class ConnectionManager:
	"""Manage connections to external systems."""
	
	def __init__(self):
		self.connectors: Dict[str, SystemConnector] = {}
		self.credentials: Dict[str, ConnectionCredentials] = {}
		self.connection_pools: Dict[str, Any] = {}
		self.health_status: Dict[str, Dict[str, Any]] = {}
	
	async def create_connector(
		self,
		connector_data: Dict[str, Any],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Create new system connector."""
		try:
			connector = SystemConnector(
				tenant_id=context.tenant_id,
				connector_name=connector_data["connector_name"],
				connector_type=ConnectorType(connector_data["connector_type"]),
				base_url=connector_data.get("base_url"),
				connection_config=connector_data.get("connection_config", {}),
				credential_id=connector_data.get("credential_id", ""),
				timeout_seconds=connector_data.get("timeout_seconds", 30),
				retry_config=connector_data.get("retry_config", {}),
				rate_limit_config=connector_data.get("rate_limit_config", {}),
				health_check_config=connector_data.get("health_check_config", {}),
				tags=connector_data.get("tags", []),
				created_by=context.user_id,
				updated_by=context.user_id
			)
			
			self.connectors[connector.connector_id] = connector
			
			# Perform initial health check
			await self._perform_health_check(connector.connector_id)
			
			logger.info(f"System connector created: {connector.connector_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="System connector created successfully",
				data={
					"connector_id": connector.connector_id,
					"connector_name": connector.connector_name,
					"connector_type": connector.connector_type.value
				}
			)
			
		except Exception as e:
			logger.error(f"Error creating system connector: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to create system connector: {e}",
				errors=[str(e)]
			)
	
	async def create_credentials(
		self,
		credentials_data: Dict[str, Any],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Create connection credentials."""
		try:
			credentials = ConnectionCredentials(
				tenant_id=context.tenant_id,
				auth_type=AuthenticationType(credentials_data["auth_type"]),
				username=credentials_data.get("username"),
				password=credentials_data.get("password"),  # Should be encrypted
				api_key=credentials_data.get("api_key"),
				bearer_token=credentials_data.get("bearer_token"),
				oauth2_config=credentials_data.get("oauth2_config", {}),
				certificate_data=credentials_data.get("certificate_data"),
				custom_headers=credentials_data.get("custom_headers", {}),
				expires_at=datetime.fromisoformat(credentials_data["expires_at"]) if credentials_data.get("expires_at") else None
			)
			
			self.credentials[credentials.credential_id] = credentials
			
			logger.info(f"Connection credentials created: {credentials.credential_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Connection credentials created successfully",
				data={
					"credential_id": credentials.credential_id,
					"auth_type": credentials.auth_type.value
				}
			)
			
		except Exception as e:
			logger.error(f"Error creating credentials: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to create credentials: {e}",
				errors=[str(e)]
			)
	
	async def test_connection(
		self,
		connector_id: str,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Test connection to external system."""
		try:
			connector = self.connectors.get(connector_id)
			if not connector:
				return WBPMServiceResponse(
					success=False,
					message="Connector not found",
					errors=["Connector not found"]
				)
			
			# Verify tenant access
			if connector.tenant_id != context.tenant_id:
				return WBPMServiceResponse(
					success=False,
					message="Access denied to connector",
					errors=["Tenant access denied"]
				)
			
			# Perform connection test
			test_result = await self._test_connector_connection(connector)
			
			return WBPMServiceResponse(
				success=test_result["success"],
				message=test_result["message"],
				data={
					"connector_id": connector_id,
					"test_timestamp": datetime.utcnow().isoformat(),
					"response_time_ms": test_result.get("response_time_ms", 0),
					"details": test_result.get("details", {})
				}
			)
			
		except Exception as e:
			logger.error(f"Error testing connection {connector_id}: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to test connection: {e}",
				errors=[str(e)]
			)
	
	async def _test_connector_connection(self, connector: SystemConnector) -> Dict[str, Any]:
		"""Test specific connector connection."""
		start_time = datetime.utcnow()
		
		try:
			if connector.connector_type == ConnectorType.REST_API:
				return await self._test_rest_api_connection(connector)
			elif connector.connector_type == ConnectorType.DATABASE:
				return await self._test_database_connection(connector)
			elif connector.connector_type == ConnectorType.EMAIL:
				return await self._test_email_connection(connector)
			else:
				return {
					"success": False,
					"message": f"Connection test not implemented for {connector.connector_type}",
					"response_time_ms": 0
				}
			
		except Exception as e:
			response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
			return {
				"success": False,
				"message": f"Connection test failed: {e}",
				"response_time_ms": response_time
			}
	
	async def _test_rest_api_connection(self, connector: SystemConnector) -> Dict[str, Any]:
		"""Test REST API connection."""
		start_time = datetime.utcnow()
		
		try:
			# Get credentials
			credentials = self.credentials.get(connector.credential_id) if connector.credential_id else None
			
			# Build request
			headers = {}
			auth = None
			
			if credentials:
				headers, auth = await self._build_auth_headers(credentials)
			
			# Determine test endpoint
			test_url = connector.base_url
			health_check_path = connector.health_check_config.get("path", "/health")
			if health_check_path:
				test_url = urljoin(test_url, health_check_path)
			
			# Make request
			async with httpx.AsyncClient(timeout=connector.timeout_seconds) as client:
				response = await client.get(test_url, headers=headers, auth=auth)
				
				response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
				
				if response.status_code < 400:
					return {
						"success": True,
						"message": "Connection successful",
						"response_time_ms": response_time,
						"details": {
							"status_code": response.status_code,
							"response_size": len(response.content)
						}
					}
				else:
					return {
						"success": False,
						"message": f"HTTP {response.status_code}: {response.text[:200]}",
						"response_time_ms": response_time,
						"details": {
							"status_code": response.status_code
						}
					}
			
		except Exception as e:
			response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
			return {
				"success": False,
				"message": f"Connection failed: {e}",
				"response_time_ms": response_time
			}
	
	async def _test_database_connection(self, connector: SystemConnector) -> Dict[str, Any]:
		"""Test database connection."""
		# In production, implement actual database connection testing
		return {
			"success": True,
			"message": "Database connection test simulated",
			"response_time_ms": 50.0
		}
	
	async def _test_email_connection(self, connector: SystemConnector) -> Dict[str, Any]:
		"""Test email connection."""
		# In production, implement actual email connection testing
		return {
			"success": True,
			"message": "Email connection test simulated",
			"response_time_ms": 100.0
		}
	
	async def _build_auth_headers(
		self,
		credentials: ConnectionCredentials
	) -> Tuple[Dict[str, str], Optional[Any]]:
		"""Build authentication headers and auth object."""
		headers = {}
		auth = None
		
		if credentials.auth_type == AuthenticationType.BASIC:
			if credentials.username and credentials.password:
				auth_string = f"{credentials.username}:{credentials.password}"
				encoded = base64.b64encode(auth_string.encode()).decode()
				headers["Authorization"] = f"Basic {encoded}"
		
		elif credentials.auth_type == AuthenticationType.BEARER_TOKEN:
			if credentials.bearer_token:
				headers["Authorization"] = f"Bearer {credentials.bearer_token}"
		
		elif credentials.auth_type == AuthenticationType.API_KEY:
			if credentials.api_key:
				# API key location depends on the service - could be header, query param, etc.
				headers["X-API-Key"] = credentials.api_key
		
		elif credentials.auth_type == AuthenticationType.CUSTOM_HEADER:
			headers.update(credentials.custom_headers)
		
		return headers, auth
	
	async def _perform_health_check(self, connector_id: str) -> None:
		"""Perform health check for connector."""
		connector = self.connectors.get(connector_id)
		if not connector:
			return
		
		try:
			test_result = await self._test_connector_connection(connector)
			
			self.health_status[connector_id] = {
				"status": "healthy" if test_result["success"] else "unhealthy",
				"last_check": datetime.utcnow(),
				"response_time_ms": test_result.get("response_time_ms", 0),
				"message": test_result["message"]
			}
			
			connector.health_status = self.health_status[connector_id]["status"]
			connector.last_health_check = datetime.utcnow()
			
		except Exception as e:
			logger.error(f"Health check failed for connector {connector_id}: {e}")
			self.health_status[connector_id] = {
				"status": "unhealthy",
				"last_check": datetime.utcnow(),
				"response_time_ms": 0,
				"message": f"Health check error: {e}"
			}


# =============================================================================
# Integration Executor
# =============================================================================

class IntegrationExecutor:
	"""Execute integration endpoints and handle data exchange."""
	
	def __init__(self, connection_manager: ConnectionManager):
		self.connection_manager = connection_manager
		self.transformation_engine = DataTransformationEngine()
		self.endpoints: Dict[str, IntegrationEndpoint] = {}
		self.execution_history: List[IntegrationExecution] = []
		self.response_cache: Dict[str, Dict[str, Any]] = {}
	
	async def create_endpoint(
		self,
		endpoint_data: Dict[str, Any],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Create new integration endpoint."""
		try:
			# Create input mappings
			input_mappings = []
			for mapping_data in endpoint_data.get("input_mappings", []):
				mapping = DataMapping(
					source_field=mapping_data["source_field"],
					target_field=mapping_data["target_field"],
					mapping_type=MappingType(mapping_data.get("mapping_type", MappingType.DIRECT)),
					transformation_expression=mapping_data.get("transformation_expression"),
					lookup_table=mapping_data.get("lookup_table", {}),
					default_value=mapping_data.get("default_value"),
					is_required=mapping_data.get("is_required", False),
					validation_rules=mapping_data.get("validation_rules", [])
				)
				input_mappings.append(mapping)
			
			# Create output mappings
			output_mappings = []
			for mapping_data in endpoint_data.get("output_mappings", []):
				mapping = DataMapping(
					source_field=mapping_data["source_field"],
					target_field=mapping_data["target_field"],
					mapping_type=MappingType(mapping_data.get("mapping_type", MappingType.DIRECT)),
					transformation_expression=mapping_data.get("transformation_expression"),
					lookup_table=mapping_data.get("lookup_table", {}),
					default_value=mapping_data.get("default_value"),
					is_required=mapping_data.get("is_required", False),
					validation_rules=mapping_data.get("validation_rules", [])
				)
				output_mappings.append(mapping)
			
			# Create endpoint
			endpoint = IntegrationEndpoint(
				tenant_id=context.tenant_id,
				endpoint_name=endpoint_data["endpoint_name"],
				connector_id=endpoint_data["connector_id"],
				endpoint_path=endpoint_data["endpoint_path"],
				http_method=endpoint_data.get("http_method", "GET"),
				input_format=DataFormat(endpoint_data.get("input_format", DataFormat.JSON)),
				output_format=DataFormat(endpoint_data.get("output_format", DataFormat.JSON)),
				input_mappings=input_mappings,
				output_mappings=output_mappings,
				request_template=endpoint_data.get("request_template"),
				response_schema=endpoint_data.get("response_schema", {}),
				error_handling=endpoint_data.get("error_handling", {}),
				cache_config=endpoint_data.get("cache_config", {}),
				tags=endpoint_data.get("tags", []),
				created_by=context.user_id,
				updated_by=context.user_id
			)
			
			self.endpoints[endpoint.endpoint_id] = endpoint
			
			logger.info(f"Integration endpoint created: {endpoint.endpoint_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Integration endpoint created successfully",
				data={
					"endpoint_id": endpoint.endpoint_id,
					"endpoint_name": endpoint.endpoint_name,
					"connector_id": endpoint.connector_id
				}
			)
			
		except Exception as e:
			logger.error(f"Error creating integration endpoint: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to create integration endpoint: {e}",
				errors=[str(e)]
			)
	
	async def execute_endpoint(
		self,
		endpoint_id: str,
		input_data: Dict[str, Any],
		context: APGTenantContext,
		execution_context: Optional[Dict[str, Any]] = None
	) -> WBPMServiceResponse:
		"""Execute integration endpoint with input data."""
		execution = IntegrationExecution(
			tenant_id=context.tenant_id,
			endpoint_id=endpoint_id,
			input_data=input_data.copy(),
			process_instance_id=execution_context.get("process_instance_id") if execution_context else None,
			task_id=execution_context.get("task_id") if execution_context else None
		)
		
		try:
			endpoint = self.endpoints.get(endpoint_id)
			if not endpoint:
				execution.execution_status = "failed"
				execution.error_message = "Endpoint not found"
				execution.completed_at = datetime.utcnow()
				return WBPMServiceResponse(
					success=False,
					message="Integration endpoint not found",
					errors=["Endpoint not found"]
				)
			
			# Verify tenant access
			if endpoint.tenant_id != context.tenant_id:
				execution.execution_status = "failed"
				execution.error_message = "Access denied"
				execution.completed_at = datetime.utcnow()
				return WBPMServiceResponse(
					success=False,
					message="Access denied to integration endpoint",
					errors=["Tenant access denied"]
				)
			
			# Check cache first
			cache_key = self._generate_cache_key(endpoint_id, input_data)
			if endpoint.cache_config.get("enabled", False):
				cached_result = self._get_cached_result(cache_key, endpoint.cache_config)
				if cached_result:
					execution.output_data = cached_result
					execution.execution_status = "success"
					execution.completed_at = datetime.utcnow()
					execution.execution_time_ms = 1.0  # Cache hit
					
					return WBPMServiceResponse(
						success=True,
						message="Integration executed successfully (cached)",
						data={
							"execution_id": execution.execution_id,
							"output_data": cached_result,
							"cached": True
						}
					)
			
			# Get connector
			connector = self.connection_manager.connectors.get(endpoint.connector_id)
			if not connector:
				execution.execution_status = "failed"
				execution.error_message = "Connector not found"
				execution.completed_at = datetime.utcnow()
				return WBPMServiceResponse(
					success=False,
					message="System connector not found",
					errors=["Connector not found"]
				)
			
			# Transform input data
			transformed_input = await self.transformation_engine.apply_mappings(
				input_data, endpoint.input_mappings, execution_context or {}
			)
			
			# Execute integration
			start_time = datetime.utcnow()
			result = await self._execute_integration(
				connector, endpoint, transformed_input, execution
			)
			
			execution.execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			execution.completed_at = datetime.utcnow()
			
			if result["success"]:
				# Transform output data
				output_data = await self.transformation_engine.apply_mappings(
					result["response_data"], endpoint.output_mappings, execution_context or {}
				)
				
				execution.output_data = output_data
				execution.execution_status = "success"
				execution.response_code = result.get("status_code")
				
				# Cache result if configured
				if endpoint.cache_config.get("enabled", False):
					self._cache_result(cache_key, output_data, endpoint.cache_config)
				
				# Store execution history
				self.execution_history.append(execution)
				
				return WBPMServiceResponse(
					success=True,
					message="Integration executed successfully",
					data={
						"execution_id": execution.execution_id,
						"output_data": output_data,
						"execution_time_ms": execution.execution_time_ms,
						"response_code": execution.response_code
					}
				)
			else:
				execution.execution_status = "failed"
				execution.error_message = result["error"]
				execution.response_code = result.get("status_code")
				
				# Store execution history
				self.execution_history.append(execution)
				
				return WBPMServiceResponse(
					success=False,
					message=f"Integration execution failed: {result['error']}",
					errors=[result["error"]]
				)
		
		except Exception as e:
			execution.execution_status = "failed"
			execution.error_message = str(e)
			execution.completed_at = datetime.utcnow()
			execution.execution_time_ms = (datetime.utcnow() - execution.started_at).total_seconds() * 1000
			
			# Store execution history
			self.execution_history.append(execution)
			
			logger.error(f"Error executing integration endpoint {endpoint_id}: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Integration execution error: {e}",
				errors=[str(e)]
			)
	
	async def _execute_integration(
		self,
		connector: SystemConnector,
		endpoint: IntegrationEndpoint,
		input_data: Dict[str, Any],
		execution: IntegrationExecution
	) -> Dict[str, Any]:
		"""Execute the actual integration call."""
		try:
			if connector.connector_type == ConnectorType.REST_API:
				return await self._execute_rest_api_call(connector, endpoint, input_data)
			elif connector.connector_type == ConnectorType.DATABASE:
				return await self._execute_database_call(connector, endpoint, input_data)
			elif connector.connector_type == ConnectorType.EMAIL:
				return await self._execute_email_call(connector, endpoint, input_data)
			else:
				return {
					"success": False,
					"error": f"Integration type {connector.connector_type} not implemented",
					"response_data": {}
				}
		
		except Exception as e:
			return {
				"success": False,
				"error": str(e),
				"response_data": {}
			}
	
	async def _execute_rest_api_call(
		self,
		connector: SystemConnector,
		endpoint: IntegrationEndpoint,
		input_data: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Execute REST API call."""
		try:
			# Get credentials
			credentials = None
			if connector.credential_id:
				credentials = self.connection_manager.credentials.get(connector.credential_id)
			
			# Build headers
			headers = {"Content-Type": "application/json"}
			auth = None
			
			if credentials:
				auth_headers, auth_obj = await self.connection_manager._build_auth_headers(credentials)
				headers.update(auth_headers)
				auth = auth_obj
			
			# Build URL
			url = urljoin(connector.base_url, endpoint.endpoint_path)
			
			# Prepare request data
			if endpoint.input_format == DataFormat.JSON:
				json_data = input_data
				data = None
			else:
				json_data = None
				data = input_data
			
			# Make request
			async with httpx.AsyncClient(timeout=connector.timeout_seconds) as client:
				if endpoint.http_method.upper() == "GET":
					response = await client.get(url, headers=headers, params=input_data, auth=auth)
				elif endpoint.http_method.upper() == "POST":
					response = await client.post(url, headers=headers, json=json_data, data=data, auth=auth)
				elif endpoint.http_method.upper() == "PUT":
					response = await client.put(url, headers=headers, json=json_data, data=data, auth=auth)
				elif endpoint.http_method.upper() == "DELETE":
					response = await client.delete(url, headers=headers, auth=auth)
				else:
					return {
						"success": False,
						"error": f"Unsupported HTTP method: {endpoint.http_method}",
						"response_data": {}
					}
				
				# Parse response
				if response.status_code < 400:
					try:
						if endpoint.output_format == DataFormat.JSON:
							response_data = response.json()
						else:
							response_data = {"content": response.text}
						
						return {
							"success": True,
							"response_data": response_data,
							"status_code": response.status_code
						}
					except Exception as e:
						return {
							"success": False,
							"error": f"Failed to parse response: {e}",
							"response_data": {"raw_content": response.text},
							"status_code": response.status_code
						}
				else:
					return {
						"success": False,
						"error": f"HTTP {response.status_code}: {response.text}",
						"response_data": {},
						"status_code": response.status_code
					}
		
		except Exception as e:
			return {
				"success": False,
				"error": f"REST API call failed: {e}",
				"response_data": {}
			}
	
	async def _execute_database_call(
		self,
		connector: SystemConnector,
		endpoint: IntegrationEndpoint,
		input_data: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Execute database call."""
		# In production, implement actual database connectivity
		return {
			"success": True,
			"response_data": {"message": "Database call simulated", "input": input_data},
			"status_code": 200
		}
	
	async def _execute_email_call(
		self,
		connector: SystemConnector,
		endpoint: IntegrationEndpoint,
		input_data: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Execute email call."""
		# In production, implement actual email sending
		return {
			"success": True,
			"response_data": {"message": "Email sent", "input": input_data},
			"status_code": 200
		}
	
	def _generate_cache_key(self, endpoint_id: str, input_data: Dict[str, Any]) -> str:
		"""Generate cache key for request."""
		data_str = json.dumps(input_data, sort_keys=True)
		hash_obj = hashlib.md5(f"{endpoint_id}:{data_str}".encode())
		return hash_obj.hexdigest()
	
	def _get_cached_result(self, cache_key: str, cache_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		"""Get cached result if valid."""
		if cache_key not in self.response_cache:
			return None
		
		cached_entry = self.response_cache[cache_key]
		ttl_seconds = cache_config.get("ttl_seconds", 300)  # Default 5 minutes
		
		if datetime.utcnow() - cached_entry["timestamp"] > timedelta(seconds=ttl_seconds):
			del self.response_cache[cache_key]
			return None
		
		return cached_entry["data"]
	
	def _cache_result(self, cache_key: str, data: Dict[str, Any], cache_config: Dict[str, Any]) -> None:
		"""Cache result."""
		self.response_cache[cache_key] = {
			"data": data,
			"timestamp": datetime.utcnow()
		}
		
		# Cleanup old cache entries (keep last 1000)
		if len(self.response_cache) > 1000:
			oldest_keys = sorted(
				self.response_cache.keys(),
				key=lambda k: self.response_cache[k]["timestamp"]
			)[:100]
			
			for key in oldest_keys:
				del self.response_cache[key]


# =============================================================================
# Integration Hub
# =============================================================================

class IntegrationHub:
	"""Main integration hub for managing all external integrations."""
	
	def __init__(self):
		self.connection_manager = ConnectionManager()
		self.integration_executor = IntegrationExecutor(self.connection_manager)
		
		# Background tasks
		self._health_check_task = None
		self._start_background_tasks()
	
	def _start_background_tasks(self) -> None:
		"""Start background tasks."""
		self._health_check_task = asyncio.create_task(self._periodic_health_checks())
	
	async def _periodic_health_checks(self) -> None:
		"""Perform periodic health checks for all connectors."""
		while True:
			try:
				for connector_id in self.connection_manager.connectors.keys():
					await self.connection_manager._perform_health_check(connector_id)
				
				# Wait 5 minutes before next check
				await asyncio.sleep(300)
				
			except Exception as e:
				logger.error(f"Error in periodic health checks: {e}")
				await asyncio.sleep(60)  # Retry in 1 minute on error
	
	async def get_integration_status(
		self,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Get overall integration hub status."""
		try:
			# Get tenant connectors
			tenant_connectors = [
				conn for conn in self.connection_manager.connectors.values()
				if conn.tenant_id == context.tenant_id
			]
			
			# Get tenant endpoints
			tenant_endpoints = [
				endpoint for endpoint in self.integration_executor.endpoints.values()
				if endpoint.tenant_id == context.tenant_id
			]
			
			# Calculate health summary
			healthy_connectors = sum(1 for conn in tenant_connectors if conn.health_status == "healthy")
			
			# Get recent execution stats
			recent_executions = [
				exec for exec in self.integration_executor.execution_history[-100:]
				if exec.tenant_id == context.tenant_id
			]
			
			success_rate = 0.0
			if recent_executions:
				successful = sum(1 for exec in recent_executions if exec.execution_status == "success")
				success_rate = (successful / len(recent_executions)) * 100
			
			return WBPMServiceResponse(
				success=True,
				message="Integration hub status retrieved successfully",
				data={
					"summary": {
						"total_connectors": len(tenant_connectors),
						"healthy_connectors": healthy_connectors,
						"total_endpoints": len(tenant_endpoints),
						"recent_executions": len(recent_executions),
						"success_rate_percentage": round(success_rate, 2)
					},
					"connectors": [
						{
							"connector_id": conn.connector_id,
							"connector_name": conn.connector_name,
							"connector_type": conn.connector_type.value,
							"health_status": conn.health_status,
							"last_health_check": conn.last_health_check.isoformat() if conn.last_health_check else None
						}
						for conn in tenant_connectors
					],
					"endpoints": [
						{
							"endpoint_id": endpoint.endpoint_id,
							"endpoint_name": endpoint.endpoint_name,
							"connector_id": endpoint.connector_id,
							"is_active": endpoint.is_active
						}
						for endpoint in tenant_endpoints
					]
				}
			)
			
		except Exception as e:
			logger.error(f"Error getting integration status: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to get integration status: {e}",
				errors=[str(e)]
			)
	
	async def shutdown(self) -> None:
		"""Shutdown integration hub and cleanup resources."""
		if self._health_check_task:
			self._health_check_task.cancel()
			try:
				await self._health_check_task
			except asyncio.CancelledError:
				pass


# =============================================================================
# Service Factory
# =============================================================================

def create_integration_hub() -> IntegrationHub:
	"""Create and configure integration hub."""
	hub = IntegrationHub()
	logger.info("Integration hub created and configured")
	return hub


# Export main classes
__all__ = [
	'IntegrationHub',
	'ConnectionManager',
	'IntegrationExecutor',
	'DataTransformationEngine',
	'SystemConnector',
	'ConnectionCredentials',
	'IntegrationEndpoint',
	'DataMapping',
	'IntegrationExecution',
	'ConnectorType',
	'AuthenticationType',
	'DataFormat',
	'MappingType',
	'create_integration_hub'
]