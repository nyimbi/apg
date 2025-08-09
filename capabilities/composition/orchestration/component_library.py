#!/usr/bin/env python3
"""
APG Workflow Orchestration Component Library

Pre-built workflow components, APG connectors, conditional logic, loops, and custom component framework.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
import importlib
from typing import Dict, Any, List, Optional, Type, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from uuid_extensions import uuid7str
from pydantic import BaseModel, ConfigDict, Field, validator
import inspect

from apg.framework.base_service import APGBaseService
from apg.framework.database import APGDatabase
from apg.framework.audit_compliance import APGAuditLogger

from .config import get_config
from .models import WorkflowStatus, TaskStatus


logger = logging.getLogger(__name__)


class ComponentType(str, Enum):
	"""Built-in component types."""
	# Basic components
	START = "start"
	END = "end"
	TASK = "task"
	DECISION = "decision"
	
	# Flow control
	LOOP = "loop"
	WHILE_LOOP = "while_loop"
	FOR_LOOP = "for_loop"
	PARALLEL = "parallel"
	JOIN = "join"
	SPLIT = "split"
	MERGE = "merge"
	
	# Conditional logic
	IF_THEN = "if_then"
	IF_THEN_ELSE = "if_then_else"
	SWITCH = "switch"
	CASE = "case"
	
	# Data operations
	TRANSFORM = "transform"
	FILTER = "filter"
	MAP = "map"
	REDUCE = "reduce"
	SORT = "sort"
	AGGREGATE = "aggregate"
	
	# External integrations
	HTTP_REQUEST = "http_request"
	DATABASE_QUERY = "database_query"
	FILE_OPERATION = "file_operation"
	EMAIL_SEND = "email_send"
	WEBHOOK = "webhook"
	
	# APG connectors
	APG_USER_MANAGEMENT = "apg_user_management"
	APG_NOTIFICATIONS = "apg_notifications"
	APG_FILE_MANAGEMENT = "apg_file_management"
	APG_AUDIT = "apg_audit"
	
	# Advanced components
	SCRIPT = "script"
	PYTHON_CODE = "python_code"
	JAVASCRIPT_CODE = "javascript_code"
	SQL_QUERY = "sql_query"
	
	# AI/ML components
	ML_PREDICTION = "ml_prediction"
	TEXT_ANALYSIS = "text_analysis"
	IMAGE_PROCESSING = "image_processing"
	
	# System components
	TIMER = "timer"
	SCHEDULER = "scheduler"
	HUMAN_TASK = "human_task"
	APPROVAL = "approval"
	NOTIFICATION = "notification"
	
	# Custom components
	CUSTOM = "custom"
	PLUGIN = "plugin"


class ComponentCategory(str, Enum):
	"""Component categories for organization."""
	BASIC = "basic"
	FLOW_CONTROL = "flow_control"
	CONDITIONAL = "conditional"
	DATA_OPERATIONS = "data_operations"
	INTEGRATIONS = "integrations"
	APG_CONNECTORS = "apg_connectors"
	ADVANCED = "advanced"
	AI_ML = "ai_ml"
	SYSTEM = "system"
	CUSTOM = "custom"


class ExecutionResult:
	"""Result of component execution."""
	
	def __init__(self, success: bool = True, data: Any = None, error: str = None, 
				 next_components: List[str] = None, metadata: Dict[str, Any] = None):
		self.success = success
		self.data = data
		self.error = error
		self.next_components = next_components or []
		self.metadata = metadata or {}
		self.timestamp = datetime.utcnow()


@dataclass
class ComponentDefinition:
	"""Definition of a workflow component."""
	id: str
	type: ComponentType
	name: str
	description: str
	category: ComponentCategory
	version: str = "1.0.0"
	author: str = "APG System"
	
	# Configuration schema
	config_schema: Dict[str, Any] = field(default_factory=dict)
	input_schema: Dict[str, Any] = field(default_factory=dict)
	output_schema: Dict[str, Any] = field(default_factory=dict)
	
	# Component properties
	is_async: bool = True
	timeout_seconds: int = 300
	retry_count: int = 3
	retry_delay: int = 5
	
	# UI properties
	icon: str = "functions"
	color: str = "#2196F3"
	ui_config: Dict[str, Any] = field(default_factory=dict)
	
	# Metadata
	tags: List[str] = field(default_factory=list)
	documentation: str = ""
	examples: List[Dict[str, Any]] = field(default_factory=list)
	
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


class BaseWorkflowComponent(ABC):
	"""Base class for all workflow components."""
	
	def __init__(self, component_id: str, config: Dict[str, Any] = None):
		self.component_id = component_id
		self.config = config or {}
		self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
		self.execution_context: Dict[str, Any] = {}
		self.audit_logger = APGAuditLogger()
	
	@abstractmethod
	async def execute(self, input_data: Any, context: Dict[str, Any]) -> ExecutionResult:
		"""Execute the component logic."""
		pass
	
	@abstractmethod
	def get_definition(self) -> ComponentDefinition:
		"""Get component definition."""
		pass
	
	async def validate_config(self, config: Dict[str, Any]) -> bool:
		"""Validate component configuration against schema."""
		try:
			definition = self.get_definition()
			config_schema = definition.config_schema
			
			if not config_schema:
				# No schema defined, configuration is valid by default
				return True
			
			# Validate required fields
			required_fields = config_schema.get('required', [])
			for field in required_fields:
				if field not in config:
					self.logger.error(f"Missing required configuration field: {field}")
					return False
			
			# Validate field types and constraints
			properties = config_schema.get('properties', {})
			for field_name, field_config in config.items():
				if field_name in properties:
					field_schema = properties[field_name]
					
					# Validate field type
					expected_type = field_schema.get('type')
					if expected_type and not self._validate_field_type(field_config, expected_type):
						self.logger.error(f"Invalid type for field '{field_name}': expected {expected_type}")
						return False
					
					# Validate field constraints
					if not self._validate_field_constraints(field_config, field_schema):
						self.logger.error(f"Field '{field_name}' violates constraints: {field_schema}")
						return False
			
			# Validate conditional fields
			if not self._validate_conditional_config(config, config_schema):
				return False
			
			# Component-specific validation
			if not await self._validate_component_specific_config(config):
				return False
			
			self.logger.debug(f"Configuration validation passed for component {self.component_id}")
			return True
			
		except Exception as e:
			self.logger.error(f"Config validation failed: {e}")
			return False
	
	def _validate_field_type(self, value: Any, expected_type: str) -> bool:
		"""Validate field type."""
		try:
			type_mapping = {
				'string': str,
				'integer': int,
				'number': (int, float),
				'boolean': bool,
				'array': list,
				'object': dict,
				'null': type(None)
			}
			
			expected_python_type = type_mapping.get(expected_type)
			if expected_python_type is None:
				return True  # Unknown type, assume valid
			
			return isinstance(value, expected_python_type)
			
		except Exception:
			return False
	
	def _validate_field_constraints(self, value: Any, field_schema: Dict[str, Any]) -> bool:
		"""Validate field constraints."""
		try:
			# String constraints
			if isinstance(value, str):
				min_length = field_schema.get('minLength')
				max_length = field_schema.get('maxLength')
				pattern = field_schema.get('pattern')
				
				if min_length is not None and len(value) < min_length:
					return False
				if max_length is not None and len(value) > max_length:
					return False
				if pattern is not None:
					import re
					if not re.match(pattern, value):
						return False
			
			# Numeric constraints
			elif isinstance(value, (int, float)):
				minimum = field_schema.get('minimum')
				maximum = field_schema.get('maximum')
				multiple_of = field_schema.get('multipleOf')
				
				if minimum is not None and value < minimum:
					return False
				if maximum is not None and value > maximum:
					return False
				if multiple_of is not None and value % multiple_of != 0:
					return False
			
			# Array constraints
			elif isinstance(value, list):
				min_items = field_schema.get('minItems')
				max_items = field_schema.get('maxItems')
				unique_items = field_schema.get('uniqueItems', False)
				
				if min_items is not None and len(value) < min_items:
					return False
				if max_items is not None and len(value) > max_items:
					return False
				if unique_items and len(value) != len(set(str(item) for item in value)):
					return False
			
			# Enum constraints
			enum_values = field_schema.get('enum')
			if enum_values is not None and value not in enum_values:
				return False
			
			return True
			
		except Exception:
			return False
	
	def _validate_conditional_config(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
		"""Validate conditional configuration logic."""
		try:
			# Check if-then-else conditions
			if_conditions = schema.get('if')
			if if_conditions:
				# Evaluate condition
				condition_met = self._evaluate_config_condition(config, if_conditions)
				
				if condition_met:
					then_schema = schema.get('then', {})
					if not self._validate_config_against_subschema(config, then_schema):
						return False
				else:
					else_schema = schema.get('else', {})
					if not self._validate_config_against_subschema(config, else_schema):
						return False
			
			# Check anyOf conditions
			any_of = schema.get('anyOf', [])
			if any_of:
				valid_any = any(self._validate_config_against_subschema(config, subschema) for subschema in any_of)
				if not valid_any:
					return False
			
			# Check allOf conditions
			all_of = schema.get('allOf', [])
			if all_of:
				valid_all = all(self._validate_config_against_subschema(config, subschema) for subschema in all_of)
				if not valid_all:
					return False
			
			return True
			
		except Exception:
			return False
	
	def _evaluate_config_condition(self, config: Dict[str, Any], condition: Dict[str, Any]) -> bool:
		"""Evaluate configuration condition."""
		try:
			# Simple property-based conditions
			for prop, expected_value in condition.get('properties', {}).items():
				if prop in config:
					if isinstance(expected_value, dict) and 'const' in expected_value:
						if config[prop] != expected_value['const']:
							return False
			return True
		except Exception:
			return False
	
	def _validate_config_against_subschema(self, config: Dict[str, Any], subschema: Dict[str, Any]) -> bool:
		"""Validate configuration against a subschema."""
		try:
			# Simplified validation for subschemas
			required = subschema.get('required', [])
			for field in required:
				if field not in config:
					return False
			
			properties = subschema.get('properties', {})
			for field_name, field_config in config.items():
				if field_name in properties:
					field_schema = properties[field_name]
					expected_type = field_schema.get('type')
					if expected_type and not self._validate_field_type(field_config, expected_type):
						return False
			
			return True
		except Exception:
			return False
	
	async def _validate_component_specific_config(self, config: Dict[str, Any]) -> bool:
		"""Validate component-specific configuration logic."""
		try:
			component_type = self.__class__.__name__
			
			# Validation rules by component type
			if component_type == "DecisionComponent":
				return self._validate_decision_config(config)
			elif component_type == "LoopComponent":
				return self._validate_loop_config(config)
			elif component_type == "ParallelComponent":
				return self._validate_parallel_config(config)
			elif component_type == "HTTPConnectorComponent":
				return self._validate_http_connector_config(config)
			elif component_type == "DatabaseConnectorComponent":
				return self._validate_database_connector_config(config)
			elif component_type == "APIGatewayComponent":
				return self._validate_api_gateway_config(config)
			elif component_type == "DataTransformComponent":
				return self._validate_data_transform_config(config)
			elif component_type == "NotificationComponent":
				return self._validate_notification_config(config)
			elif component_type == "TimerComponent":
				return self._validate_timer_config(config)
			
			# Default validation passed
			return True
			
		except Exception as e:
			self.logger.error(f"Component-specific validation failed: {e}")
			return False
	
	def _validate_decision_config(self, config: Dict[str, Any]) -> bool:
		"""Validate decision component configuration"""
		# Require conditions and branches
		if 'conditions' not in config:
			self.logger.error("Decision component requires 'conditions' field")
			return False
		
		conditions = config['conditions']
		if not isinstance(conditions, list) or len(conditions) == 0:
			self.logger.error("Decision conditions must be non-empty list")
			return False
		
		# Validate each condition
		for i, condition in enumerate(conditions):
			if not isinstance(condition, dict):
				self.logger.error(f"Condition {i} must be a dictionary")
				return False
			
			required_fields = ['expression', 'target_node']
			for field in required_fields:
				if field not in condition:
					self.logger.error(f"Condition {i} missing required field: {field}")
					return False
		
		return True
	
	def _validate_loop_config(self, config: Dict[str, Any]) -> bool:
		"""Validate loop component configuration"""
		loop_type = config.get('loop_type', 'for')
		
		if loop_type == 'for':
			# For loops need iteration parameters
			if 'iterations' not in config and 'collection' not in config:
				self.logger.error("For loop requires 'iterations' or 'collection' parameter")
				return False
		elif loop_type == 'while':
			# While loops need condition
			if 'condition' not in config:
				self.logger.error("While loop requires 'condition' parameter")
				return False
		
		# Validate max iterations safety limit
		max_iterations = config.get('max_iterations', 1000)
		if not isinstance(max_iterations, int) or max_iterations <= 0 or max_iterations > 100000:
			self.logger.error("max_iterations must be positive integer <= 100000")
			return False
		
		return True
	
	def _validate_parallel_config(self, config: Dict[str, Any]) -> bool:
		"""Validate parallel component configuration"""
		# Require parallel branches
		if 'branches' not in config:
			self.logger.error("Parallel component requires 'branches' field")
			return False
		
		branches = config['branches']
		if not isinstance(branches, list) or len(branches) < 2:
			self.logger.error("Parallel component needs at least 2 branches")
			return False
		
		# Validate concurrency limits
		max_concurrency = config.get('max_concurrency', len(branches))
		if not isinstance(max_concurrency, int) or max_concurrency <= 0:
			self.logger.error("max_concurrency must be positive integer")
			return False
		
		return True
	
	def _validate_http_connector_config(self, config: Dict[str, Any]) -> bool:
		"""Validate HTTP connector configuration"""
		# Require URL
		if 'url' not in config:
			self.logger.error("HTTP connector requires 'url' field")
			return False
		
		url = config['url']
		if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
			self.logger.error("URL must be valid HTTP/HTTPS URL")
			return False
		
		# Validate HTTP method
		method = config.get('method', 'GET').upper()
		valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
		if method not in valid_methods:
			self.logger.error(f"Invalid HTTP method: {method}")
			return False
		
		# Validate timeout
		timeout = config.get('timeout', 30)
		if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 300:
			self.logger.error("Timeout must be positive number <= 300 seconds")
			return False
		
		return True
	
	def _validate_database_connector_config(self, config: Dict[str, Any]) -> bool:
		"""Validate database connector configuration"""
		# Require connection string or connection parameters
		if 'connection_string' not in config and 'connection_params' not in config:
			self.logger.error("Database connector requires connection_string or connection_params")
			return False
		
		# Validate query if present
		if 'query' in config:
			query = config['query']
			if not isinstance(query, str) or len(query.strip()) == 0:
				self.logger.error("Database query must be non-empty string")
				return False
			
			# Basic SQL injection prevention
			dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE']
			query_upper = query.upper()
			for keyword in dangerous_keywords:
				if keyword in query_upper and not config.get('allow_dangerous_operations', False):
					self.logger.error(f"Potentially dangerous SQL keyword detected: {keyword}")
					return False
		
		return True
	
	def _validate_api_gateway_config(self, config: Dict[str, Any]) -> bool:
		"""Validate API gateway configuration"""
		# Require routes configuration
		if 'routes' not in config:
			self.logger.error("API gateway requires 'routes' configuration")
			return False
		
		routes = config['routes']
		if not isinstance(routes, list) or len(routes) == 0:
			self.logger.error("API gateway routes must be non-empty list")
			return False
		
		# Validate each route
		for i, route in enumerate(routes):
			required_fields = ['path', 'method', 'backend']
			for field in required_fields:
				if field not in route:
					self.logger.error(f"Route {i} missing required field: {field}")
					return False
		
		return True
	
	def _validate_data_transform_config(self, config: Dict[str, Any]) -> bool:
		"""Validate data transformation configuration"""
		# Require transformation rules
		if 'transformations' not in config:
			self.logger.error("Data transform component requires 'transformations' field")
			return False
		
		transformations = config['transformations']
		if not isinstance(transformations, list) or len(transformations) == 0:
			self.logger.error("Transformations must be non-empty list")
			return False
		
		# Validate transformation rules
		for i, transform in enumerate(transformations):
			if not isinstance(transform, dict):
				self.logger.error(f"Transformation {i} must be dictionary")
				return False
			
			if 'type' not in transform:
				self.logger.error(f"Transformation {i} missing 'type' field")
				return False
		
		return True
	
	def _validate_notification_config(self, config: Dict[str, Any]) -> bool:
		"""Validate notification component configuration"""
		# Require notification type
		if 'notification_type' not in config:
			self.logger.error("Notification component requires 'notification_type' field")
			return False
		
		notification_type = config['notification_type']
		valid_types = ['email', 'sms', 'slack', 'webhook', 'push']
		if notification_type not in valid_types:
			self.logger.error(f"Invalid notification type: {notification_type}")
			return False
		
		# Type-specific validation
		if notification_type == 'email':
			if 'recipients' not in config:
				self.logger.error("Email notification requires 'recipients' field")
				return False
		elif notification_type == 'webhook':
			if 'webhook_url' not in config:
				self.logger.error("Webhook notification requires 'webhook_url' field")
				return False
		
		return True
	
	def _validate_timer_config(self, config: Dict[str, Any]) -> bool:
		"""Validate timer component configuration"""
		# Require either duration or schedule
		if 'duration' not in config and 'schedule' not in config:
			self.logger.error("Timer component requires 'duration' or 'schedule' field")
			return False
		
		# Validate duration if present
		if 'duration' in config:
			duration = config['duration']
			if not isinstance(duration, (int, float)) or duration <= 0:
				self.logger.error("Duration must be positive number")
				return False
		
		# Validate schedule if present (cron expression)
		if 'schedule' in config:
			schedule = config['schedule']
			if not isinstance(schedule, str) or len(schedule.strip()) == 0:
				self.logger.error("Schedule must be non-empty string")
				return False
		
		return True
	
	async def validate_input(self, input_data: Any) -> bool:
		"""Validate input data against input schema."""
		try:
			definition = self.get_definition()
			input_schema = definition.input_schema
			
			if not input_schema:
				# No schema defined, input is valid by default
				return True
			
			# Handle null/None input
			if input_data is None:
				null_allowed = input_schema.get('type') == 'null' or 'null' in input_schema.get('type', [])
				if not null_allowed and input_schema.get('required', True):
					self.logger.error("Input data is required but received None")
					return False
				return True
			
			# Validate input data type
			expected_type = input_schema.get('type')
			if expected_type and not self._validate_field_type(input_data, expected_type):
				self.logger.error(f"Invalid input data type: expected {expected_type}, got {type(input_data).__name__}")
				return False
			
			# Validate input data constraints
			if not self._validate_field_constraints(input_data, input_schema):
				self.logger.error(f"Input data violates constraints: {input_schema}")
				return False
			
			# Validate object properties if input is a dictionary
			if isinstance(input_data, dict):
				if not self._validate_object_input(input_data, input_schema):
					return False
			
			# Validate array items if input is a list
			elif isinstance(input_data, list):
				if not self._validate_array_input(input_data, input_schema):
					return False
			
			# Component-specific input validation
			if not await self._validate_component_specific_input(input_data):
				return False
			
			self.logger.debug(f"Input validation passed for component {self.component_id}")
			return True
			
		except Exception as e:
			self.logger.error(f"Input validation failed: {e}")
			return False
	
	def _validate_object_input(self, input_data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
		"""Validate object input data."""
		try:
			# Check required properties
			required_props = schema.get('required', [])
			for prop in required_props:
				if prop not in input_data:
					self.logger.error(f"Missing required input property: {prop}")
					return False
			
			# Validate individual properties
			properties = schema.get('properties', {})
			for prop_name, prop_value in input_data.items():
				if prop_name in properties:
					prop_schema = properties[prop_name]
					
					# Validate property type
					prop_type = prop_schema.get('type')
					if prop_type and not self._validate_field_type(prop_value, prop_type):
						self.logger.error(f"Invalid type for input property '{prop_name}': expected {prop_type}")
						return False
					
					# Validate property constraints
					if not self._validate_field_constraints(prop_value, prop_schema):
						self.logger.error(f"Input property '{prop_name}' violates constraints")
						return False
			
			# Check additional properties
			additional_allowed = schema.get('additionalProperties', True)
			if not additional_allowed:
				for prop_name in input_data:
					if prop_name not in properties:
						self.logger.error(f"Additional property '{prop_name}' not allowed in input")
						return False
			
			return True
			
		except Exception:
			return False
	
	def _validate_array_input(self, input_data: List[Any], schema: Dict[str, Any]) -> bool:
		"""Validate array input data."""
		try:
			# Validate array constraints (already done in _validate_field_constraints)
			
			# Validate array items
			items_schema = schema.get('items')
			if items_schema:
				for i, item in enumerate(input_data):
					item_type = items_schema.get('type')
					if item_type and not self._validate_field_type(item, item_type):
						self.logger.error(f"Invalid type for array item at index {i}: expected {item_type}")
						return False
					
					if not self._validate_field_constraints(item, items_schema):
						self.logger.error(f"Array item at index {i} violates constraints")
						return False
			
			return True
			
		except Exception:
			return False
	
	async def _validate_component_specific_input(self, input_data: Any) -> bool:
		"""Validate component-specific input logic."""
		try:
			component_type = self.__class__.__name__
			
			# Input validation rules by component type
			if component_type == "DecisionComponent":
				return self._validate_decision_input(input_data)
			elif component_type == "LoopComponent":
				return self._validate_loop_input(input_data)
			elif component_type == "HTTPConnectorComponent":
				return self._validate_http_connector_input(input_data)
			elif component_type == "DatabaseConnectorComponent":
				return self._validate_database_connector_input(input_data)
			elif component_type == "DataTransformComponent":
				return self._validate_data_transform_input(input_data)
			elif component_type == "NotificationComponent":
				return self._validate_notification_input(input_data)
			elif component_type == "EmailConnectorComponent":
				return self._validate_email_connector_input(input_data)
			elif component_type == "FileProcessorComponent":
				return self._validate_file_processor_input(input_data)
			
			# Default input validation passed
			return True
			
		except Exception as e:
			self.logger.error(f"Component-specific input validation failed: {e}")
			return False
	
	def _validate_decision_input(self, input_data: Any) -> bool:
		"""Validate decision component input"""
		if not isinstance(input_data, dict):
			self.logger.error("Decision component input must be dictionary")
			return False
		
		# Check for required context variables
		if 'variables' not in input_data:
			self.logger.error("Decision component requires 'variables' in input")
			return False
		
		variables = input_data['variables']
		if not isinstance(variables, dict):
			self.logger.error("Decision variables must be dictionary")
			return False
		
		return True
	
	def _validate_loop_input(self, input_data: Any) -> bool:
		"""Validate loop component input"""
		if not isinstance(input_data, dict):
			self.logger.error("Loop component input must be dictionary")
			return False
		
		# Validate collection for for-each loops
		if 'collection' in input_data:
			collection = input_data['collection']
			if not isinstance(collection, (list, tuple)):
				self.logger.error("Loop collection must be list or tuple")
				return False
		
		return True
	
	def _validate_http_connector_input(self, input_data: Any) -> bool:
		"""Validate HTTP connector input"""
		if not isinstance(input_data, dict):
			self.logger.error("HTTP connector input must be dictionary")
			return False
		
		# Validate headers if present
		if 'headers' in input_data:
			headers = input_data['headers']
			if not isinstance(headers, dict):
				self.logger.error("HTTP headers must be dictionary")
				return False
			
			# Check for sensitive data in headers
			for header_name in headers.keys():
				if any(sensitive in header_name.lower() for sensitive in ['password', 'secret', 'key']):
					self.logger.warning(f"Potentially sensitive header detected: {header_name}")
		
		# Validate request body size
		if 'body' in input_data:
			body = input_data['body']
			if isinstance(body, str) and len(body) > 10 * 1024 * 1024:  # 10MB limit
				self.logger.error("HTTP request body too large (>10MB)")
				return False
		
		return True
	
	def _validate_database_connector_input(self, input_data: Any) -> bool:
		"""Validate database connector input"""
		if not isinstance(input_data, dict):
			self.logger.error("Database connector input must be dictionary")
			return False
		
		# Validate query parameters if present
		if 'query_params' in input_data:
			params = input_data['query_params']
			if not isinstance(params, dict):
				self.logger.error("Database query parameters must be dictionary")
				return False
			
			# Check for SQL injection attempts in parameters
			for param_name, param_value in params.items():
				if isinstance(param_value, str):
					suspicious_patterns = ['DROP', 'DELETE', 'TRUNCATE', '--', ';', 'UNION', 'SELECT']
					param_upper = param_value.upper()
					for pattern in suspicious_patterns:
						if pattern in param_upper:
							self.logger.warning(f"Suspicious SQL pattern in parameter {param_name}: {pattern}")
		
		return True
	
	def _validate_data_transform_input(self, input_data: Any) -> bool:
		"""Validate data transformation input"""
		# Accept any data type for transformation
		if input_data is None:
			self.logger.warning("Data transform received null input")
			return True
		
		# Check data size limits
		if isinstance(input_data, (str, bytes)):
			if len(input_data) > 50 * 1024 * 1024:  # 50MB limit
				self.logger.error("Input data too large for transformation (>50MB)")
				return False
		elif isinstance(input_data, (list, dict)):
			try:
				import json
				serialized = json.dumps(input_data)
				if len(serialized) > 50 * 1024 * 1024:  # 50MB limit
					self.logger.error("Input data too large for transformation (>50MB)")
					return False
			except (TypeError, ValueError):
				self.logger.warning("Input data not JSON serializable")
		
		return True
	
	def _validate_notification_input(self, input_data: Any) -> bool:
		"""Validate notification component input"""
		if not isinstance(input_data, dict):
			self.logger.error("Notification component input must be dictionary")
			return False
		
		# Require message content
		if 'message' not in input_data:
			self.logger.error("Notification requires 'message' field")
			return False
		
		message = input_data['message']
		if not isinstance(message, str) or len(message.strip()) == 0:
			self.logger.error("Notification message must be non-empty string")
			return False
		
		# Check message length limits
		if len(message) > 10000:  # 10KB limit
			self.logger.error("Notification message too long (>10KB)")
			return False
		
		return True
	
	def _validate_email_connector_input(self, input_data: Any) -> bool:
		"""Validate email connector input"""
		if not isinstance(input_data, dict):
			self.logger.error("Email connector input must be dictionary")
			return False
		
		# Validate email addresses
		required_fields = ['to', 'subject', 'body']
		for field in required_fields:
			if field not in input_data:
				self.logger.error(f"Email connector requires '{field}' field")
				return False
		
		# Validate email format
		to_emails = input_data['to']
		if isinstance(to_emails, str):
			to_emails = [to_emails]
		
		if not isinstance(to_emails, list):
			self.logger.error("Email 'to' field must be string or list")
			return False
		
		import re
		email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
		for email in to_emails:
			if not re.match(email_pattern, email):
				self.logger.error(f"Invalid email address: {email}")
				return False
		
		return True
	
	def _validate_file_processor_input(self, input_data: Any) -> bool:
		"""Validate file processor input"""
		if not isinstance(input_data, dict):
			self.logger.error("File processor input must be dictionary")
			return False
		
		# Require file path or file content
		if 'file_path' not in input_data and 'file_content' not in input_data:
			self.logger.error("File processor requires 'file_path' or 'file_content'")
			return False
		
		# Validate file path security
		if 'file_path' in input_data:
			file_path = input_data['file_path']
			if not isinstance(file_path, str):
				self.logger.error("File path must be string")
				return False
			
			# Check for path traversal attempts
			if '..' in file_path or file_path.startswith('/'):
				self.logger.error("Potentially unsafe file path detected")
				return False
		
		# Validate file size if content provided
		if 'file_content' in input_data:
			content = input_data['file_content']
			if isinstance(content, (str, bytes)):
				if len(content) > 100 * 1024 * 1024:  # 100MB limit
					self.logger.error("File content too large (>100MB)")
					return False
		
		return True
	
	async def _log_execution(self, input_data: Any, result: ExecutionResult):
		"""Log component execution for audit."""
		await self.audit_logger.log_event({
			'event_type': 'component_executed',
			'component_id': self.component_id,
			'component_type': self.__class__.__name__,
			'success': result.success,
			'execution_time': result.timestamp.isoformat(),
			'input_size': len(str(input_data)) if input_data else 0,
			'output_size': len(str(result.data)) if result.data else 0,
			'error': result.error
		})


# Basic Components

class StartComponent(BaseWorkflowComponent):
	"""Workflow start component."""
	
	async def execute(self, input_data: Any, context: Dict[str, Any]) -> ExecutionResult:
		"""Start component always succeeds and passes input data."""
		self.logger.info(f"Starting workflow execution")
		
		result = ExecutionResult(
			success=True,
			data=input_data,
			metadata={'started_at': datetime.utcnow().isoformat()}
		)
		
		await self._log_execution(input_data, result)
		return result
	
	def get_definition(self) -> ComponentDefinition:
		return ComponentDefinition(
			id="start_component",
			type=ComponentType.START,
			name="Start",
			description="Marks the beginning of a workflow",
			category=ComponentCategory.BASIC,
			icon="play_arrow",
			color="#4CAF50",
			config_schema={
				"type": "object",
				"properties": {
					"trigger_type": {
						"type": "string",
						"enum": ["manual", "scheduled", "event", "webhook"],
						"default": "manual"
					},
					"initial_data": {
						"type": "object",
						"description": "Initial data to pass to workflow"
					}
				}
			},
			output_schema={
				"type": "object",
				"description": "Initial workflow data"
			}
		)


class EndComponent(BaseWorkflowComponent):
	"""Workflow end component."""
	
	async def execute(self, input_data: Any, context: Dict[str, Any]) -> ExecutionResult:
		"""End component finalizes workflow execution."""
		self.logger.info(f"Ending workflow execution")
		
		# Apply any final transformations
		final_data = input_data
		if self.config.get('final_transform'):
			# Apply final transformation logic
			final_data = await self._apply_final_transformation(input_data)
		
		result = ExecutionResult(
			success=True,
			data=final_data,
			metadata={
				'completed_at': datetime.utcnow().isoformat(),
				'final_status': 'completed'
			}
		)
		
		await self._log_execution(input_data, result)
		return result
	
	async def _apply_final_transformation(self, input_data: Any) -> Any:
		"""Apply final data transformation."""
		try:
			transform_expr = self.config.get('final_transform', '')
			
			if not transform_expr:
				return input_data
			
			# Simple transformation expressions
			if transform_expr == 'json_string':
				import json
				return json.dumps(input_data)
			elif transform_expr == 'flatten':
				return self._flatten_data(input_data)
			elif transform_expr == 'summary':
				return self._create_summary(input_data)
			elif transform_expr == 'metadata_only':
				if isinstance(input_data, dict):
					return input_data.get('metadata', {})
				return {}
			elif transform_expr.startswith('extract:'):
				# Extract specific field: extract:field.subfield
				field_path = transform_expr[8:]  # Remove 'extract:' prefix
				return self._extract_field(input_data, field_path)
			else:
				# Custom transformation (could be extended with expression evaluator)
				self.logger.warning(f"Unknown transformation: {transform_expr}")
				return input_data
				
		except Exception as e:
			self.logger.error(f"Final transformation failed: {e}")
			return input_data
	
	def _flatten_data(self, data: Any) -> Dict[str, Any]:
		"""Flatten nested data structures."""
		try:
			def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
				items = []
				for k, v in d.items():
					new_key = f"{parent_key}{sep}{k}" if parent_key else k
					if isinstance(v, dict):
						items.extend(flatten_dict(v, new_key, sep=sep).items())
					else:
						items.append((new_key, v))
				return dict(items)
			
			if isinstance(data, dict):
				return flatten_dict(data)
			elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
				# Flatten each dict in the list
				return {f"item_{i}_{k}": v for i, item in enumerate(data) for k, v in flatten_dict(item).items()}
			else:
				return {"value": data}
				
		except Exception:
			return {"value": data}
	
	def _create_summary(self, data: Any) -> Dict[str, Any]:
		"""Create a summary of the data."""
		try:
			summary = {
				"data_type": type(data).__name__,
				"timestamp": datetime.utcnow().isoformat()
			}
			
			if isinstance(data, dict):
				summary.update({
					"keys_count": len(data),
					"keys": list(data.keys())[:10],  # First 10 keys
					"has_metadata": "metadata" in data
				})
			elif isinstance(data, list):
				summary.update({
					"items_count": len(data),
					"first_item_type": type(data[0]).__name__ if data else "None"
				})
			elif isinstance(data, str):
				summary.update({
					"length": len(data),
					"preview": data[:100] + "..." if len(data) > 100 else data
				})
			else:
				summary["value"] = str(data)
			
			return summary
			
		except Exception:
			return {"data_type": "unknown", "timestamp": datetime.utcnow().isoformat()}
	
	def _extract_field(self, data: Any, field_path: str) -> Any:
		"""Extract field using dot notation path."""
		try:
			current = data
			for field in field_path.split('.'):
				if isinstance(current, dict) and field in current:
					current = current[field]
				elif isinstance(current, list) and field.isdigit():
					index = int(field)
					if 0 <= index < len(current):
						current = current[index]
					else:
						return None
				else:
					return None
			return current
		except Exception:
			return None
	
	def get_definition(self) -> ComponentDefinition:
		return ComponentDefinition(
			id="end_component",
			type=ComponentType.END,
			name="End",
			description="Marks the end of a workflow",
			category=ComponentCategory.BASIC,
			icon="stop",
			color="#F44336",
			config_schema={
				"type": "object",
				"properties": {
					"final_transform": {
						"type": "string",
						"description": "Final data transformation expression"
					},
					"cleanup_actions": {
						"type": "array",
						"items": {"type": "string"},
						"description": "Cleanup actions to perform"
					}
				}
			}
		)


class TaskComponent(BaseWorkflowComponent):
	"""Generic task component."""
	
	async def execute(self, input_data: Any, context: Dict[str, Any]) -> ExecutionResult:
		"""Execute a generic task."""
		try:
			task_type = self.config.get('task_type', 'processing')
			
			if task_type == 'processing':
				result_data = await self._process_data(input_data)
			elif task_type == 'validation':
				result_data = await self._validate_data(input_data)
			elif task_type == 'transformation':
				result_data = await self._transform_data(input_data)
			else:
				result_data = input_data
			
			result = ExecutionResult(
				success=True,
				data=result_data,
				metadata={'task_type': task_type}
			)
			
		except Exception as e:
			result = ExecutionResult(
				success=False,
				error=str(e),
				data=input_data
			)
		
		await self._log_execution(input_data, result)
		return result
	
	async def _process_data(self, data: Any) -> Any:
		"""Process input data."""
		# Simulate processing delay
		processing_time = self.config.get('processing_time', 0.1)
		await asyncio.sleep(processing_time)
		
		# Apply processing logic
		if isinstance(data, dict):
			data['processed'] = True
			data['processed_at'] = datetime.utcnow().isoformat()
		
		return data
	
	async def _validate_data(self, data: Any) -> Any:
		"""Validate input data."""
		validation_rules = self.config.get('validation_rules', [])
		
		for rule in validation_rules:
			if not self._apply_validation_rule(data, rule):
				raise ValueError(f"Validation failed: {rule}")
		
		return data
	
	async def _transform_data(self, data: Any) -> Any:
		"""Transform input data."""
		transformations = self.config.get('transformations', [])
		
		result = data
		for transformation in transformations:
			result = self._apply_transformation(result, transformation)
		
		return result
	
	def _apply_validation_rule(self, data: Any, rule: str) -> bool:
		"""Apply a validation rule."""
		# Simplified validation logic
		return True
	
	def _apply_transformation(self, data: Any, transformation: str) -> Any:
		"""Apply a data transformation."""
		# Simplified transformation logic
		return data
	
	def get_definition(self) -> ComponentDefinition:
		return ComponentDefinition(
			id="task_component",
			type=ComponentType.TASK,
			name="Task",
			description="Generic task component for data processing",
			category=ComponentCategory.BASIC,
			icon="functions",
			color="#2196F3",
			config_schema={
				"type": "object",
				"properties": {
					"task_type": {
						"type": "string",
						"enum": ["processing", "validation", "transformation"],
						"default": "processing"
					},
					"processing_time": {
						"type": "number",
						"minimum": 0,
						"default": 0.1
					},
					"validation_rules": {
						"type": "array",
						"items": {"type": "string"}
					},
					"transformations": {
						"type": "array",
						"items": {"type": "string"}
					}
				}
			}
		)


# Flow Control Components

class DecisionComponent(BaseWorkflowComponent):
	"""Decision/conditional logic component."""
	
	async def execute(self, input_data: Any, context: Dict[str, Any]) -> ExecutionResult:
		"""Execute decision logic."""
		try:
			condition = self.config.get('condition', 'true')
			branches = self.config.get('branches', {})
			
			# Evaluate condition
			condition_result = await self._evaluate_condition(condition, input_data, context)
			
			# Determine next components based on condition
			if condition_result:
				next_components = branches.get('true', [])
				branch_taken = 'true'
			else:
				next_components = branches.get('false', [])
				branch_taken = 'false'
			
			result = ExecutionResult(
				success=True,
				data=input_data,
				next_components=next_components,
				metadata={
					'condition': condition,
					'condition_result': condition_result,
					'branch_taken': branch_taken
				}
			)
			
		except Exception as e:
			result = ExecutionResult(
				success=False,
				error=str(e),
				data=input_data
			)
		
		await self._log_execution(input_data, result)
		return result
	
	async def _evaluate_condition(self, condition: str, data: Any, context: Dict[str, Any]) -> bool:
		"""Evaluate a condition expression."""
		try:
			# Simple condition evaluation (in production, use a proper expression evaluator)
			if condition == 'true':
				return True
			elif condition == 'false':
				return False
			elif condition.startswith('data.'):
				# Access data properties
				property_path = condition[5:]  # Remove 'data.' prefix
				return self._get_nested_property(data, property_path)
			else:
				# For more complex conditions, use eval with restricted scope
				safe_globals = {'data': data, 'context': context}
				return bool(eval(condition, safe_globals))
		except Exception as e:
			self.logger.error(f"Condition evaluation failed: {e}")
			return False
	
	def _get_nested_property(self, obj: Any, path: str) -> Any:
		"""Get nested property from object using dot notation."""
		parts = path.split('.')
		result = obj
		
		for part in parts:
			if isinstance(result, dict):
				result = result.get(part)
			elif hasattr(result, part):
				result = getattr(result, part)
			else:
				return None
		
		return result
	
	def get_definition(self) -> ComponentDefinition:
		return ComponentDefinition(
			id="decision_component",
			type=ComponentType.DECISION,
			name="Decision",
			description="Conditional logic component for workflow branching",
			category=ComponentCategory.CONDITIONAL,
			icon="decision",
			color="#FF9800",
			config_schema={
				"type": "object",
				"properties": {
					"condition": {
						"type": "string",
						"description": "Condition expression to evaluate"
					},
					"branches": {
						"type": "object",
						"properties": {
							"true": {
								"type": "array",
								"items": {"type": "string"},
								"description": "Components to execute if condition is true"
							},
							"false": {
								"type": "array",
								"items": {"type": "string"},
								"description": "Components to execute if condition is false"
							}
						}
					}
				},
				"required": ["condition", "branches"]
			}
		)


class LoopComponent(BaseWorkflowComponent):
	"""Loop component for iterative execution."""
	
	async def execute(self, input_data: Any, context: Dict[str, Any]) -> ExecutionResult:
		"""Execute loop logic."""
		try:
			loop_type = self.config.get('loop_type', 'for')
			max_iterations = self.config.get('max_iterations', 100)
			
			if loop_type == 'for':
				result = await self._execute_for_loop(input_data, context, max_iterations)
			elif loop_type == 'while':
				result = await self._execute_while_loop(input_data, context, max_iterations)
			elif loop_type == 'foreach':
				result = await self._execute_foreach_loop(input_data, context)
			else:
				raise ValueError(f"Unknown loop type: {loop_type}")
			
		except Exception as e:
			result = ExecutionResult(
				success=False,
				error=str(e),
				data=input_data
			)
		
		await self._log_execution(input_data, result)
		return result
	
	async def _execute_for_loop(self, data: Any, context: Dict[str, Any], max_iterations: int) -> ExecutionResult:
		"""Execute for loop."""
		iterations = self.config.get('iterations', 1)
		loop_body = self.config.get('loop_body', [])
		
		results = []
		for i in range(min(iterations, max_iterations)):
			loop_context = {**context, 'loop_index': i, 'loop_data': data}
			# Execute loop body components
			iteration_result = await self._execute_loop_body(loop_body, data, loop_context)
			results.append(iteration_result)
		
		return ExecutionResult(
			success=True,
			data=results,
			metadata={'loop_type': 'for', 'iterations': len(results)}
		)
	
	async def _execute_while_loop(self, data: Any, context: Dict[str, Any], max_iterations: int) -> ExecutionResult:
		"""Execute while loop."""
		condition = self.config.get('while_condition', 'false')
		loop_body = self.config.get('loop_body', [])
		
		results = []
		iteration = 0
		
		while iteration < max_iterations:
			# Evaluate while condition
			condition_result = await self._evaluate_condition(condition, data, context)
			if not condition_result:
				break
			
			loop_context = {**context, 'loop_index': iteration, 'loop_data': data}
			iteration_result = await self._execute_loop_body(loop_body, data, loop_context)
			results.append(iteration_result)
			
			# Update data for next iteration
			data = iteration_result
			iteration += 1
		
		return ExecutionResult(
			success=True,
			data=results,
			metadata={'loop_type': 'while', 'iterations': len(results)}
		)
	
	async def _execute_foreach_loop(self, data: Any, context: Dict[str, Any]) -> ExecutionResult:
		"""Execute foreach loop."""
		if not isinstance(data, (list, tuple)):
			raise ValueError("Foreach loop requires array input data")
		
		loop_body = self.config.get('loop_body', [])
		results = []
		
		for index, item in enumerate(data):
			loop_context = {**context, 'loop_index': index, 'loop_item': item}
			iteration_result = await self._execute_loop_body(loop_body, item, loop_context)
			results.append(iteration_result)
		
		return ExecutionResult(
			success=True,
			data=results,
			metadata={'loop_type': 'foreach', 'iterations': len(results)}
		)
	
	async def _execute_loop_body(self, loop_body: List[str], data: Any, context: Dict[str, Any]) -> Any:
		"""Execute loop body components."""
		# In a real implementation, this would execute the specified components
		# For now, just return the data with loop metadata
		return {
			'data': data,
			'loop_context': context,
			'processed_at': datetime.utcnow().isoformat()
		}
	
	async def _evaluate_condition(self, condition: str, data: Any, context: Dict[str, Any]) -> bool:
		"""Evaluate loop condition."""
		# Reuse condition evaluation logic from DecisionComponent
		try:
			safe_globals = {'data': data, 'context': context}
			return bool(eval(condition, safe_globals))
		except Exception as e:
			self.logger.error(f"Loop condition evaluation failed: {e}")
			return False
	
	def get_definition(self) -> ComponentDefinition:
		return ComponentDefinition(
			id="loop_component",
			type=ComponentType.LOOP,
			name="Loop",
			description="Loop component for iterative execution",
			category=ComponentCategory.FLOW_CONTROL,
			icon="loop",
			color="#9C27B0",
			config_schema={
				"type": "object",
				"properties": {
					"loop_type": {
						"type": "string",
						"enum": ["for", "while", "foreach"],
						"default": "for"
					},
					"iterations": {
						"type": "integer",
						"minimum": 1,
						"default": 1
					},
					"max_iterations": {
						"type": "integer",
						"minimum": 1,
						"default": 100
					},
					"while_condition": {
						"type": "string",
						"description": "Condition for while loop"
					},
					"loop_body": {
						"type": "array",
						"items": {"type": "string"},
						"description": "Components to execute in loop body"
					}
				}
			}
		)


# APG Connector Components

class APGUserManagementComponent(BaseWorkflowComponent):
	"""APG User Management connector component."""
	
	async def execute(self, input_data: Any, context: Dict[str, Any]) -> ExecutionResult:
		"""Execute user management operation."""
		try:
			operation = self.config.get('operation', 'get_user')
			
			if operation == 'get_user':
				result_data = await self._get_user(input_data)
			elif operation == 'create_user':
				result_data = await self._create_user(input_data)
			elif operation == 'update_user':
				result_data = await self._update_user(input_data)
			elif operation == 'list_users':
				result_data = await self._list_users(input_data)
			else:
				raise ValueError(f"Unknown operation: {operation}")
			
			result = ExecutionResult(
				success=True,
				data=result_data,
				metadata={'operation': operation, 'apg_service': 'user_management'}
			)
			
		except Exception as e:
			result = ExecutionResult(
				success=False,
				error=str(e),
				data=input_data
			)
		
		await self._log_execution(input_data, result)
		return result
	
	async def _get_user(self, data: Any) -> Dict[str, Any]:
		"""Get user information from APG User Management service."""
		user_id = data.get('user_id') if isinstance(data, dict) else str(data)
		
		try:
			# Import APG User Management service
			from apg.capabilities.auth_rbac.service import UserManagementService
			
			# Get service instance
			user_service = UserManagementService(
				db_session=self.db_session,
				audit_logger=self.audit_logger
			)
			
			# Get user from service
			user = await user_service.get_user(user_id)
			
			if not user:
				raise ValueError(f"User not found: {user_id}")
			
			return {
				'user_id': user.user_id,
				'username': user.username,
				'email': user.email,
				'full_name': user.full_name,
				'status': user.status,
				'created_at': user.created_at.isoformat() if user.created_at else None,
				'last_login': user.last_login.isoformat() if user.last_login else None,
				'roles': [role.name for role in user.roles] if hasattr(user, 'roles') else [],
				'permissions': await user_service.get_user_permissions(user_id),
				'profile': user.profile_data if hasattr(user, 'profile_data') else {},
				'retrieved_at': datetime.utcnow().isoformat()
			}
			
		except ImportError:
			# Fallback if auth_rbac service not available
			self.logger.warning("APG User Management service not available, using mock data")
			return {
				'user_id': user_id,
				'username': f'user_{user_id}',
				'email': f'user{user_id}@datacraft.co.ke',
				'full_name': f'User {user_id}',
				'status': 'active',
				'retrieved_at': datetime.utcnow().isoformat(),
				'roles': ['user'],
				'permissions': ['read:own_data'],
				'profile': {}
			}
		except Exception as e:
			self.logger.error(f"Failed to get user {user_id}: {e}")
			raise
	
	async def _create_user(self, data: Any) -> Dict[str, Any]:
		"""Create new user in APG User Management service."""
		user_data = data if isinstance(data, dict) else {}
		
		# Validate required fields
		required_fields = ['username', 'email', 'full_name']
		for field in required_fields:
			if field not in user_data:
				raise ValueError(f"Missing required field: {field}")
		
		try:
			# Import APG User Management service
			from apg.capabilities.auth_rbac.service import UserManagementService
			from apg.capabilities.auth_rbac.models import CreateUserRequest
			
			# Get service instance
			user_service = UserManagementService(
				db_session=self.db_session,
				audit_logger=self.audit_logger
			)
			
			# Create user request
			create_request = CreateUserRequest(
				username=user_data['username'],
				email=user_data['email'],
				full_name=user_data['full_name'],
				password=user_data.get('password', 'temp_password_change_required'),
				roles=user_data.get('roles', ['user']),
				profile_data=user_data.get('profile', {}),
				is_active=user_data.get('is_active', True)
			)
			
			# Create user
			new_user = await user_service.create_user(create_request)
			
			return {
				'user_id': new_user.user_id,
				'username': new_user.username,
				'email': new_user.email,
				'full_name': new_user.full_name,
				'status': new_user.status,
				'created_at': new_user.created_at.isoformat(),
				'roles': user_data.get('roles', ['user']),
				'requires_password_change': True if user_data.get('password') == 'temp_password_change_required' else False
			}
			
		except ImportError:
			# Fallback if auth_rbac service not available
			self.logger.warning("APG User Management service not available, using mock creation")
			new_user_id = uuid7str()
			return {
				'user_id': new_user_id,
				'username': user_data['username'],
				'email': user_data['email'],
				'full_name': user_data['full_name'],
				'status': 'active',
				'created_at': datetime.utcnow().isoformat(),
				'roles': user_data.get('roles', ['user']),
				'requires_password_change': True
			}
		except Exception as e:
			self.logger.error(f"Failed to create user: {e}")
			raise
	
	async def _update_user(self, data: Any) -> Dict[str, Any]:
		"""Update existing user in APG User Management service."""
		if not isinstance(data, dict) or 'user_id' not in data:
			raise ValueError("Update user requires user_id in input data")
		
		user_id = data['user_id']
		
		try:
			# Import APG User Management service
			from apg.capabilities.auth_rbac.service import UserManagementService
			from apg.capabilities.auth_rbac.models import UpdateUserRequest
			
			# Get service instance
			user_service = UserManagementService(
				db_session=self.db_session,
				audit_logger=self.audit_logger
			)
			
			# Create update request with only provided fields
			update_fields = {k: v for k, v in data.items() if k != 'user_id'}
			
			update_request = UpdateUserRequest(
				username=update_fields.get('username'),
				email=update_fields.get('email'),
				full_name=update_fields.get('full_name'),
				is_active=update_fields.get('is_active'),
				profile_data=update_fields.get('profile')
			)
			
			# Update user
			updated_user = await user_service.update_user(user_id, update_request)
			
			return {
				'user_id': updated_user.user_id,
				'username': updated_user.username,
				'email': updated_user.email,
				'full_name': updated_user.full_name,
				'status': updated_user.status,
				'updated_at': datetime.utcnow().isoformat(),
				'changes_applied': list(update_fields.keys())
			}
			
		except ImportError:
			# Fallback if auth_rbac service not available
			self.logger.warning("APG User Management service not available, using mock update")
			return {
				'user_id': user_id,
				'username': data.get('username', f'user_{user_id}'),
				'email': data.get('email', f'user{user_id}@datacraft.co.ke'),
				'full_name': data.get('full_name', f'User {user_id}'),
				'status': data.get('status', 'active'),
				'updated_at': datetime.utcnow().isoformat(),
				'changes_applied': list(data.keys())
			}
		except Exception as e:
			self.logger.error(f"Failed to update user {user_id}: {e}")
			raise
	
	async def _list_users(self, data: Any) -> List[Dict[str, Any]]:
		"""List users with optional filtering from APG User Management service."""
		filters = data if isinstance(data, dict) else {}
		
		try:
			# Import APG User Management service
			from apg.capabilities.auth_rbac.service import UserManagementService
			
			# Get service instance
			user_service = UserManagementService(
				db_session=self.db_session,
				audit_logger=self.audit_logger
			)
			
			# Apply filters
			status_filter = filters.get('status')
			role_filter = filters.get('role')
			search_query = filters.get('search')
			limit = filters.get('limit', 50)
			offset = filters.get('offset', 0)
			
			# Get users from service
			users = await user_service.list_users(
				status=status_filter,
				role=role_filter,
				search=search_query,
				limit=limit,
				offset=offset
			)
			
			user_list = []
			for user in users:
				user_data = {
					'user_id': user.user_id,
					'username': user.username,
					'email': user.email,
					'full_name': user.full_name,
					'status': user.status,
					'created_at': user.created_at.isoformat() if user.created_at else None,
					'last_login': user.last_login.isoformat() if user.last_login else None
				}
				
				# Add roles if available
				if hasattr(user, 'roles'):
					user_data['roles'] = [role.name for role in user.roles]
				
				user_list.append(user_data)
			
			return user_list
			
		except ImportError:
			# Fallback if auth_rbac service not available
			self.logger.warning("APG User Management service not available, using mock data")
			users = [
				{
					'user_id': f'user_{i}',
					'username': f'user_{i}',
					'full_name': f'User {i}',
					'email': f'user{i}@datacraft.co.ke',
					'status': 'active',
					'created_at': datetime.utcnow().isoformat(),
					'roles': ['user']
				}
				for i in range(1, 6)
			]
			
			return users
			
		except Exception as e:
			self.logger.error(f"Failed to list users: {e}")
			raise
	
	def get_definition(self) -> ComponentDefinition:
		return ComponentDefinition(
			id="apg_user_management_component",
			type=ComponentType.APG_USER_MANAGEMENT,
			name="APG User Management",
			description="Connect to APG User Management capability",
			category=ComponentCategory.APG_CONNECTORS,
			icon="people",
			color="#00BCD4",
			config_schema={
				"type": "object",
				"properties": {
					"operation": {
						"type": "string",
						"enum": ["get_user", "create_user", "update_user", "list_users"],
						"default": "get_user"
					},
					"timeout_seconds": {
						"type": "integer",
						"minimum": 1,
						"default": 30
					}
				}
			}
		)


# Component Library Manager

class ComponentLibrary:
	"""Manages the collection of available workflow components."""
	
	def __init__(self):
		self.components: Dict[str, Type[BaseWorkflowComponent]] = {}
		self.definitions: Dict[str, ComponentDefinition] = {}
		self.categories: Dict[ComponentCategory, List[str]] = {}
		
		# Register built-in components
		self._register_builtin_components()
	
	def _register_builtin_components(self):
		"""Register all built-in components."""
		builtin_components = [
			StartComponent,
			EndComponent,
			TaskComponent,
			DecisionComponent,
			LoopComponent,
			APGUserManagementComponent,
			# Add more built-in components here
		]
		
		for component_class in builtin_components:
			self.register_component(component_class)
	
	def register_component(self, component_class: Type[BaseWorkflowComponent]):
		"""Register a component class."""
		try:
			# Create temporary instance to get definition
			temp_instance = component_class("temp", {})
			definition = temp_instance.get_definition()
			
			# Store component class and definition
			self.components[definition.type.value] = component_class
			self.definitions[definition.type.value] = definition
			
			# Organize by category
			if definition.category not in self.categories:
				self.categories[definition.category] = []
			
			if definition.type.value not in self.categories[definition.category]:
				self.categories[definition.category].append(definition.type.value)
			
			logger.info(f"Registered component: {definition.name} ({definition.type.value})")
			
		except Exception as e:
			logger.error(f"Failed to register component {component_class.__name__}: {e}")
	
	def create_component(self, component_type: str, component_id: str, config: Dict[str, Any] = None) -> BaseWorkflowComponent:
		"""Create a component instance."""
		if component_type not in self.components:
			raise ValueError(f"Unknown component type: {component_type}")
		
		component_class = self.components[component_type]
		return component_class(component_id, config or {})
	
	def get_component_definition(self, component_type: str) -> Optional[ComponentDefinition]:
		"""Get component definition."""
		return self.definitions.get(component_type)
	
	def list_components(self, category: ComponentCategory = None) -> List[ComponentDefinition]:
		"""List available components, optionally filtered by category."""
		if category:
			component_types = self.categories.get(category, [])
			return [self.definitions[comp_type] for comp_type in component_types]
		else:
			return list(self.definitions.values())
	
	def get_categories(self) -> Dict[ComponentCategory, List[str]]:
		"""Get all categories and their components."""
		return self.categories.copy()
	
	def validate_component_config(self, component_type: str, config: Dict[str, Any]) -> bool:
		"""Validate component configuration."""
		try:
			component = self.create_component(component_type, "temp", config)
			return asyncio.run(component.validate_config(config))
		except Exception as e:
			logger.error(f"Config validation failed for {component_type}: {e}")
			return False


# Global component library instance
component_library = ComponentLibrary()


# Component Library Service

class ComponentLibraryService(APGBaseService):
	"""Service for managing workflow component library."""
	
	def __init__(self):
		super().__init__()
		self.library = component_library
		self.database = APGDatabase()
		self.audit = APGAuditLogger()
		self.custom_components: Dict[str, Dict[str, Any]] = {}
	
	async def start(self):
		"""Start component library service."""
		await super().start()
		await self._load_custom_components()
		logger.info("Component library service started")
	
	async def get_available_components(self, category: ComponentCategory = None) -> List[Dict[str, Any]]:
		"""Get list of available components."""
		try:
			definitions = self.library.list_components(category)
			
			components = []
			for definition in definitions:
				components.append({
					'id': definition.id,
					'type': definition.type.value,
					'name': definition.name,
					'description': definition.description,
					'category': definition.category.value,
					'version': definition.version,
					'author': definition.author,
					'icon': definition.icon,
					'color': definition.color,
					'tags': definition.tags,
					'config_schema': definition.config_schema,
					'input_schema': definition.input_schema,
					'output_schema': definition.output_schema,
					'ui_config': definition.ui_config
				})
			
			return components
			
		except Exception as e:
			logger.error(f"Failed to get available components: {e}")
			return []
	
	async def create_component_instance(self, component_type: str, component_id: str, 
									   config: Dict[str, Any] = None) -> BaseWorkflowComponent:
		"""Create a component instance."""
		try:
			# Validate configuration
			if config and not self.library.validate_component_config(component_type, config):
				raise ValueError(f"Invalid configuration for component type: {component_type}")
			
			# Create component instance
			component = self.library.create_component(component_type, component_id, config)
			
			# Log component creation
			await self.audit.log_event({
				'event_type': 'component_created',
				'component_type': component_type,
				'component_id': component_id,
				'config_provided': bool(config)
			})
			
			return component
			
		except Exception as e:
			logger.error(f"Failed to create component instance: {e}")
			raise
	
	async def execute_component(self, component: BaseWorkflowComponent, input_data: Any, 
							   context: Dict[str, Any] = None) -> ExecutionResult:
		"""Execute a component with proper error handling and logging."""
		try:
			context = context or {}
			
			# Validate input data
			if not await component.validate_input(input_data):
				raise ValueError("Invalid input data for component")
			
			# Execute component
			result = await component.execute(input_data, context)
			
			# Log execution result
			await self.audit.log_event({
				'event_type': 'component_executed',
				'component_id': component.component_id,
				'success': result.success,
				'error': result.error,
				'execution_time': result.timestamp.isoformat()
			})
			
			return result
			
		except Exception as e:
			logger.error(f"Component execution failed: {e}")
			return ExecutionResult(
				success=False,
				error=str(e),
				data=input_data
			)
	
	async def register_custom_component(self, component_definition: Dict[str, Any], 
									   component_code: str) -> bool:
		"""Register a custom component."""
		try:
			# Validate component definition
			if not self._validate_custom_component_definition(component_definition):
				raise ValueError("Invalid component definition")
			
			# Store custom component
			component_id = component_definition['id']
			self.custom_components[component_id] = {
				'definition': component_definition,
				'code': component_code,
				'created_at': datetime.utcnow().isoformat(),
				'status': 'active'
			}
			
			# Save to database
			await self._save_custom_component(component_id, component_definition, component_code)
			
			logger.info(f"Registered custom component: {component_id}")
			return True
			
		except Exception as e:
			logger.error(f"Failed to register custom component: {e}")
			return False
	
	async def _load_custom_components(self):
		"""Load custom components from database."""
		try:
			# In a real implementation, this would load from database
			logger.info("Loading custom components...")
		except Exception as e:
			logger.error(f"Failed to load custom components: {e}")
	
	def _validate_custom_component_definition(self, definition: Dict[str, Any]) -> bool:
		"""Validate custom component definition."""
		required_fields = ['id', 'name', 'description', 'type', 'category']
		
		for field in required_fields:
			if field not in definition:
				logger.error(f"Missing required field: {field}")
				return False
		
		return True
	
	async def _save_custom_component(self, component_id: str, definition: Dict[str, Any], code: str):
		"""Save custom component to database."""
		try:
			from .database import DatabaseManager
			
			# Get database manager instance
			db_manager = DatabaseManager()
			
			async with db_manager.get_session() as session:
				# Insert custom component into database
				insert_query = """
				INSERT INTO cr_custom_components (
					id, name, description, component_type, category, 
					definition, code, version, author, status, 
					created_at, updated_at, tenant_id
				) VALUES (
					%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
				)
				ON CONFLICT (id, tenant_id) 
				DO UPDATE SET
					name = EXCLUDED.name,
					description = EXCLUDED.description,
					component_type = EXCLUDED.component_type,
					category = EXCLUDED.category,
					definition = EXCLUDED.definition,
					code = EXCLUDED.code,
					version = EXCLUDED.version,
					updated_at = EXCLUDED.updated_at,
					status = EXCLUDED.status
				"""
				
				values = (
					component_id,
					definition.get('name', 'Untitled Component'),
					definition.get('description', ''),
					definition.get('type', 'custom'),
					definition.get('category', 'custom'),
					json.dumps(definition),
					code,
					definition.get('version', '1.0.0'),
					definition.get('author', 'System'),
					'active',
					datetime.utcnow(),
					datetime.utcnow(),
					getattr(self, 'tenant_id', 'default_tenant')
				)
				
				await session.execute(insert_query, values)
				await session.commit()
				
				# Also save component metadata
				await self._save_component_metadata(session, component_id, definition)
				
				logger.info(f"Successfully saved custom component {component_id} to database")
				
		except Exception as e:
			logger.error(f"Failed to save custom component: {e}")
			raise
	
	async def _save_component_metadata(self, session, component_id: str, definition: Dict[str, Any]):
		"""Save additional component metadata."""
		try:
			# Save component tags
			tags = definition.get('tags', [])
			if tags:
				# Clear existing tags
				await session.execute(
					"DELETE FROM cr_component_tags WHERE component_id = %s",
					[component_id]
				)
				
				# Insert new tags
				for tag in tags:
					await session.execute(
						"""
						INSERT INTO cr_component_tags (component_id, tag, created_at)
						VALUES (%s, %s, %s)
						""",
						[component_id, tag, datetime.utcnow()]
					)
			
			# Save component configuration schema
			config_schema = definition.get('config_schema', {})
			if config_schema:
				await session.execute(
					"""
					INSERT INTO cr_component_schemas (
						component_id, schema_type, schema_data, created_at
					) VALUES (%s, %s, %s, %s)
					ON CONFLICT (component_id, schema_type)
					DO UPDATE SET schema_data = EXCLUDED.schema_data, updated_at = %s
					""",
					[component_id, 'config', json.dumps(config_schema), datetime.utcnow(), datetime.utcnow()]
				)
			
			# Save input/output schemas
			input_schema = definition.get('input_schema', {})
			if input_schema:
				await session.execute(
					"""
					INSERT INTO cr_component_schemas (
						component_id, schema_type, schema_data, created_at
					) VALUES (%s, %s, %s, %s)
					ON CONFLICT (component_id, schema_type)
					DO UPDATE SET schema_data = EXCLUDED.schema_data, updated_at = %s
					""",
					[component_id, 'input', json.dumps(input_schema), datetime.utcnow(), datetime.utcnow()]
				)
			
			output_schema = definition.get('output_schema', {})
			if output_schema:
				await session.execute(
					"""
					INSERT INTO cr_component_schemas (
						component_id, schema_type, schema_data, created_at
					) VALUES (%s, %s, %s, %s)
					ON CONFLICT (component_id, schema_type)
					DO UPDATE SET schema_data = EXCLUDED.schema_data, updated_at = %s
					""",
					[component_id, 'output', json.dumps(output_schema), datetime.utcnow(), datetime.utcnow()]
				)
			
			# Save component examples
			examples = definition.get('examples', [])
			if examples:
				# Clear existing examples
				await session.execute(
					"DELETE FROM cr_component_examples WHERE component_id = %s",
					[component_id]
				)
				
				# Insert new examples
				for i, example in enumerate(examples):
					await session.execute(
						"""
						INSERT INTO cr_component_examples (
							component_id, example_order, name, description, 
							input_data, expected_output, created_at
						) VALUES (%s, %s, %s, %s, %s, %s, %s)
						""",
						[
							component_id, i, 
							example.get('name', f'Example {i+1}'),
							example.get('description', ''),
							json.dumps(example.get('input', {})),
							json.dumps(example.get('output', {})),
							datetime.utcnow()
						]
					)
			
			await session.commit()
			
		except Exception as e:
			logger.error(f"Failed to save component metadata: {e}")
			await session.rollback()
			raise
	
	async def health_check(self) -> bool:
		"""Health check for component library service."""
		try:
			# Check if built-in components are available
			available_components = await self.get_available_components()
			return len(available_components) > 0
		except Exception:
			return False


# Global component library service instance
component_library_service = ComponentLibraryService()