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
			# Override in subclasses for component-specific validation
			return True
		except Exception:
			return False
	
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
			# Override in subclasses for component-specific input validation
			return True
		except Exception:
			return False
	
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
		"""Get user information."""
		user_id = data.get('user_id') if isinstance(data, dict) else str(data)
		
		# In real implementation, this would call APG User Management API
		return {
			'user_id': user_id,
			'name': f'User {user_id}',
			'email': f'user{user_id}@example.com',
			'status': 'active',
			'retrieved_at': datetime.utcnow().isoformat()
		}
	
	async def _create_user(self, data: Any) -> Dict[str, Any]:
		"""Create new user."""
		user_data = data if isinstance(data, dict) else {}
		
		# Validate required fields
		required_fields = ['name', 'email']
		for field in required_fields:
			if field not in user_data:
				raise ValueError(f"Missing required field: {field}")
		
		# In real implementation, this would call APG User Management API
		new_user = {
			'user_id': uuid7str(),
			'name': user_data['name'],
			'email': user_data['email'],
			'status': 'active',
			'created_at': datetime.utcnow().isoformat()
		}
		
		return new_user
	
	async def _update_user(self, data: Any) -> Dict[str, Any]:
		"""Update existing user."""
		if not isinstance(data, dict) or 'user_id' not in data:
			raise ValueError("Update user requires user_id in input data")
		
		# In real implementation, this would call APG User Management API
		updated_user = {
			**data,
			'updated_at': datetime.utcnow().isoformat()
		}
		
		return updated_user
	
	async def _list_users(self, data: Any) -> List[Dict[str, Any]]:
		"""List users with optional filtering."""
		filters = data if isinstance(data, dict) else {}
		
		# In real implementation, this would call APG User Management API
		users = [
			{
				'user_id': f'user_{i}',
				'name': f'User {i}',
				'email': f'user{i}@example.com',
				'status': 'active'
			}
			for i in range(1, 6)
		]
		
		return users
	
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