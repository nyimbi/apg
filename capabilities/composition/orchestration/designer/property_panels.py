"""
APG Workflow Property Panels Manager

Dynamic property panel system for component configuration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from uuid import uuid4
import json

from pydantic import BaseModel, Field, ConfigDict
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

logger = logging.getLogger(__name__)

class PropertyDefinition(BaseModel):
	"""Definition of a property for UI rendering."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Basic info
	name: str = Field(..., description="Property name")
	type: str = Field(..., description="Property type")
	label: str = Field(..., description="Display label")
	description: str = Field(default="", description="Property description")
	
	# Value
	value: Any = Field(default=None, description="Current value")
	default_value: Any = Field(default=None, description="Default value")
	
	# Validation
	required: bool = Field(default=False, description="Whether property is required")
	validation: Optional[Dict[str, Any]] = Field(default=None, description="Validation rules")
	
	# UI rendering
	widget: str = Field(default="input", description="UI widget type")
	widget_config: Dict[str, Any] = Field(default_factory=dict, description="Widget-specific configuration")
	
	# Organization
	group: str = Field(default="general", description="Property group")
	order: int = Field(default=0, description="Display order within group")
	
	# State
	visible: bool = Field(default=True, description="Visibility state")
	enabled: bool = Field(default=True, description="Enabled state")
	
	# Dependencies
	depends_on: Optional[List[str]] = Field(default=None, description="Dependent properties")
	condition: Optional[Dict[str, Any]] = Field(default=None, description="Visibility/enablement condition")

class PropertyGroup(BaseModel):
	"""Group of related properties."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(..., description="Group ID")
	label: str = Field(..., description="Group display label")
	description: str = Field(default="", description="Group description")
	icon: Optional[str] = Field(default=None, description="Group icon")
	collapsible: bool = Field(default=True, description="Whether group can be collapsed")
	collapsed: bool = Field(default=False, description="Initial collapsed state")
	order: int = Field(default=0, description="Display order")

class PropertyPanelState(BaseModel):
	"""State of a property panel."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	component_id: str = Field(..., description="Associated component ID")
	component_type: str = Field(..., description="Component type")
	properties: List[PropertyDefinition] = Field(default_factory=list, description="Property definitions")
	groups: List[PropertyGroup] = Field(default_factory=list, description="Property groups")
	values: Dict[str, Any] = Field(default_factory=dict, description="Current property values")
	errors: Dict[str, List[str]] = Field(default_factory=dict, description="Validation errors")
	touched: Dict[str, bool] = Field(default_factory=dict, description="Touched state")

class PropertyPanelManager:
	"""
	Dynamic property panel manager for component configuration.
	
	Features:
	- Dynamic property generation based on component definitions
	- Rich widget library (inputs, selects, code editors, etc.)
	- Real-time validation
	- Conditional visibility and enablement
	- Property grouping and organization
	- Custom property types and widgets
	"""
	
	def __init__(self, config):
		self.config = config
		self.panel_states: Dict[str, PropertyPanelState] = {}
		self.widget_registry: Dict[str, Dict[str, Any]] = {}
		self.is_initialized = False
		
		logger.info("Property panel manager initialized")
	
	async def initialize(self) -> None:
		"""Initialize the property panel manager."""
		try:
			# Register built-in widgets
			await self._register_builtin_widgets()
			
			self.is_initialized = True
			logger.info("Property panel manager initialization completed")
			
		except Exception as e:
			logger.error(f"Failed to initialize property panel manager: {e}")
			raise
	
	async def shutdown(self) -> None:
		"""Shutdown the property panel manager."""
		try:
			self.panel_states.clear()
			self.widget_registry.clear()
			self.is_initialized = False
			logger.info("Property panel manager shutdown completed")
		except Exception as e:
			logger.error(f"Error during property panel manager shutdown: {e}")
	
	async def create_property_panel(self, session_id: str, component_id: str, component_type: str, component_definition: Dict[str, Any]) -> PropertyPanelState:
		"""Create a property panel for a component."""
		try:
			panel_key = f"{session_id}_{component_id}"
			
			# Generate properties from component definition
			properties = await self._generate_properties(component_definition)
			groups = await self._generate_groups(properties)
			
			# Create panel state
			panel_state = PropertyPanelState(
				component_id=component_id,
				component_type=component_type,
				properties=properties,
				groups=groups,
				values=self._extract_default_values(properties)
			)
			
			self.panel_states[panel_key] = panel_state
			
			logger.debug(f"Created property panel for component {component_id}")
			return panel_state
			
		except Exception as e:
			logger.error(f"Failed to create property panel: {e}")
			raise
	
	async def update_property_value(self, session_id: str, component_id: str, property_name: str, value: Any) -> Dict[str, Any]:
		"""Update a property value and validate."""
		try:
			panel_key = f"{session_id}_{component_id}"
			
			if panel_key not in self.panel_states:
				raise ValueError(f"Property panel not found for component {component_id}")
			
			panel_state = self.panel_states[panel_key]
			
			# Find property definition
			property_def = next((p for p in panel_state.properties if p.name == property_name), None)
			if not property_def:
				raise ValueError(f"Property {property_name} not found")
			
			# Update value
			panel_state.values[property_name] = value
			panel_state.touched[property_name] = True
			property_def.value = value
			
			# Validate property
			validation_result = await self._validate_property(property_def, value)
			
			if validation_result['valid']:
				# Clear errors
				if property_name in panel_state.errors:
					del panel_state.errors[property_name]
			else:
				# Set errors
				panel_state.errors[property_name] = validation_result['errors']
			
			# Update dependent properties
			await self._update_dependent_properties(panel_state, property_name)
			
			return {
				'valid': validation_result['valid'],
				'errors': validation_result['errors'],
				'panel_state': panel_state.model_dump()
			}
			
		except Exception as e:
			logger.error(f"Failed to update property value: {e}")
			raise
	
	async def get_property_panel(self, session_id: str, component_id: str) -> Optional[PropertyPanelState]:
		"""Get property panel state."""
		try:
			panel_key = f"{session_id}_{component_id}"
			return self.panel_states.get(panel_key)
		except Exception as e:
			logger.error(f"Failed to get property panel: {e}")
			return None
	
	async def validate_all_properties(self, session_id: str, component_id: str) -> Dict[str, Any]:
		"""Validate all properties in a panel."""
		try:
			panel_key = f"{session_id}_{component_id}"
			
			if panel_key not in self.panel_states:
				raise ValueError(f"Property panel not found for component {component_id}")
			
			panel_state = self.panel_states[panel_key]
			all_errors = {}
			
			# Validate each property
			for property_def in panel_state.properties:
				value = panel_state.values.get(property_def.name, property_def.default_value)
				validation_result = await self._validate_property(property_def, value)
				
				if not validation_result['valid']:
					all_errors[property_def.name] = validation_result['errors']
			
			# Update panel state
			panel_state.errors = all_errors
			
			return {
				'valid': len(all_errors) == 0,
				'errors': all_errors,
				'panel_state': panel_state.model_dump()
			}
			
		except Exception as e:
			logger.error(f"Failed to validate all properties: {e}")
			raise
	
	async def get_property_values(self, session_id: str, component_id: str) -> Dict[str, Any]:
		"""Get all property values for a component."""
		try:
			panel_key = f"{session_id}_{component_id}"
			
			if panel_key not in self.panel_states:
				return {}
			
			panel_state = self.panel_states[panel_key]
			return panel_state.values.copy()
			
		except Exception as e:
			logger.error(f"Failed to get property values: {e}")
			return {}
	
	async def reset_property_panel(self, session_id: str, component_id: str) -> None:
		"""Reset property panel to default values."""
		try:
			panel_key = f"{session_id}_{component_id}"
			
			if panel_key not in self.panel_states:
				return
			
			panel_state = self.panel_states[panel_key]
			
			# Reset to default values
			panel_state.values = self._extract_default_values(panel_state.properties)
			panel_state.errors.clear()
			panel_state.touched.clear()
			
			# Update property values
			for property_def in panel_state.properties:
				property_def.value = panel_state.values.get(property_def.name, property_def.default_value)
			
		except Exception as e:
			logger.error(f"Failed to reset property panel: {e}")
	
	async def register_custom_widget(self, widget_type: str, widget_config: Dict[str, Any]) -> None:
		"""Register a custom property widget."""
		try:
			self.widget_registry[widget_type] = widget_config
			logger.info(f"Registered custom widget: {widget_type}")
		except Exception as e:
			logger.error(f"Failed to register custom widget: {e}")
			raise
	
	async def get_widget_config(self, widget_type: str) -> Optional[Dict[str, Any]]:
		"""Get configuration for a widget type."""
		return self.widget_registry.get(widget_type)
	
	# Private methods
	
	async def _generate_properties(self, component_definition: Dict[str, Any]) -> List[PropertyDefinition]:
		"""Generate property definitions from component definition."""
		try:
			properties = []
			component_properties = component_definition.get('properties', [])
			
			for prop_data in component_properties:
				# Create property definition
				property_def = PropertyDefinition(
					name=prop_data.get('name'),
					type=prop_data.get('type', 'string'),
					label=prop_data.get('label', prop_data.get('name', '').title()),
					description=prop_data.get('description', ''),
					default_value=prop_data.get('default_value'),
					required=prop_data.get('required', False),
					validation=prop_data.get('validation'),
					widget=prop_data.get('widget', self._get_default_widget(prop_data.get('type', 'string'))),
					widget_config=prop_data.get('widget_config', {}),
					group=prop_data.get('group', 'general'),
					order=prop_data.get('order', 0),
					value=prop_data.get('default_value')
				)
				
				# Add widget-specific configuration
				await self._configure_widget(property_def, prop_data)
				
				properties.append(property_def)
			
			# Sort by group and order
			return sorted(properties, key=lambda p: (p.group, p.order, p.name))
			
		except Exception as e:
			logger.error(f"Failed to generate properties: {e}")
			return []
	
	async def _generate_groups(self, properties: List[PropertyDefinition]) -> List[PropertyGroup]:
		"""Generate property groups from properties."""
		try:
			groups_dict = {}
			
			for prop in properties:
				if prop.group not in groups_dict:
					groups_dict[prop.group] = PropertyGroup(
						id=prop.group,
						label=prop.group.replace('_', ' ').title(),
						order=self._get_group_order(prop.group)
					)
			
			return sorted(groups_dict.values(), key=lambda g: (g.order, g.label))
			
		except Exception as e:
			logger.error(f"Failed to generate groups: {e}")
			return []
	
	def _extract_default_values(self, properties: List[PropertyDefinition]) -> Dict[str, Any]:
		"""Extract default values from properties."""
		values = {}
		for prop in properties:
			if prop.default_value is not None:
				values[prop.name] = prop.default_value
		return values
	
	def _get_default_widget(self, property_type: str) -> str:
		"""Get default widget for property type."""
		widget_mapping = {
			'string': 'input',
			'number': 'number',
			'integer': 'number',
			'boolean': 'checkbox',
			'array': 'array',
			'object': 'json',
			'code': 'code'
		}
		return widget_mapping.get(property_type, 'input')
	
	def _get_group_order(self, group_name: str) -> int:
		"""Get display order for group."""
		group_orders = {
			'general': 1,
			'configuration': 2,
			'advanced': 3,
			'validation': 4,
			'display': 5,
			'behavior': 6
		}
		return group_orders.get(group_name, 999)
	
	async def _configure_widget(self, property_def: PropertyDefinition, prop_data: Dict[str, Any]) -> None:
		"""Configure widget-specific settings."""
		try:
			widget_type = property_def.widget
			
			if widget_type == 'select':
				# Configure select options
				options = prop_data.get('options', [])
				property_def.widget_config['options'] = options
			
			elif widget_type == 'code':
				# Configure code editor
				property_def.widget_config.update({
					'language': prop_data.get('language', 'javascript'),
					'theme': 'vs-dark',
					'height': prop_data.get('height', 200)
				})
			
			elif widget_type == 'number':
				# Configure number input
				property_def.widget_config.update({
					'min': prop_data.get('min'),
					'max': prop_data.get('max'),
					'step': prop_data.get('step', 1)
				})
			
			elif widget_type == 'array':
				# Configure array editor
				property_def.widget_config.update({
					'item_type': prop_data.get('item_type', 'string'),
					'min_items': prop_data.get('min_items', 0),
					'max_items': prop_data.get('max_items')
				})
			
		except Exception as e:
			logger.error(f"Failed to configure widget: {e}")
	
	async def _validate_property(self, property_def: PropertyDefinition, value: Any) -> Dict[str, Any]:
		"""Validate a property value."""
		try:
			errors = []
			
			# Required validation
			if property_def.required and (value is None or value == ''):
				errors.append(f"{property_def.label} is required")
			
			# Type validation
			if value is not None:
				type_validation = await self._validate_property_type(property_def, value)
				errors.extend(type_validation)
			
			# Custom validation rules
			if property_def.validation and value is not None:
				custom_validation = await self._validate_custom_rules(property_def, value)
				errors.extend(custom_validation)
			
			return {
				'valid': len(errors) == 0,
				'errors': errors
			}
			
		except Exception as e:
			logger.error(f"Failed to validate property: {e}")
			return {
				'valid': False,
				'errors': [f"Validation error: {e}"]
			}
	
	async def _validate_property_type(self, property_def: PropertyDefinition, value: Any) -> List[str]:
		"""Validate property type."""
		errors = []
		
		try:
			if property_def.type == 'number' or property_def.type == 'integer':
				try:
					float(value)
					if property_def.type == 'integer' and not isinstance(value, int):
						if float(value) != int(float(value)):
							errors.append(f"{property_def.label} must be an integer")
				except (ValueError, TypeError):
					errors.append(f"{property_def.label} must be a number")
			
			elif property_def.type == 'boolean':
				if not isinstance(value, bool) and value not in ['true', 'false', 0, 1]:
					errors.append(f"{property_def.label} must be true or false")
			
			elif property_def.type == 'array':
				if not isinstance(value, list):
					errors.append(f"{property_def.label} must be an array")
			
			elif property_def.type == 'object':
				if not isinstance(value, dict):
					# Try to parse as JSON string
					if isinstance(value, str):
						try:
							json.loads(value)
						except json.JSONDecodeError:
							errors.append(f"{property_def.label} must be valid JSON")
					else:
						errors.append(f"{property_def.label} must be an object")
			
		except Exception as e:
			errors.append(f"Type validation error: {e}")
		
		return errors
	
	async def _validate_custom_rules(self, property_def: PropertyDefinition, value: Any) -> List[str]:
		"""Validate custom validation rules."""
		errors = []
		validation_rules = property_def.validation or {}
		
		try:
			# Min/max validation
			if 'min' in validation_rules:
				if isinstance(value, (int, float)) and value < validation_rules['min']:
					errors.append(f"{property_def.label} must be at least {validation_rules['min']}")
				elif isinstance(value, str) and len(value) < validation_rules['min']:
					errors.append(f"{property_def.label} must be at least {validation_rules['min']} characters")
			
			if 'max' in validation_rules:
				if isinstance(value, (int, float)) and value > validation_rules['max']:
					errors.append(f"{property_def.label} must be at most {validation_rules['max']}")
				elif isinstance(value, str) and len(value) > validation_rules['max']:
					errors.append(f"{property_def.label} must be at most {validation_rules['max']} characters")
			
			# Pattern validation
			if 'pattern' in validation_rules and isinstance(value, str):
				import re
				pattern = validation_rules['pattern']
				if not re.match(pattern, value):
					errors.append(f"{property_def.label} does not match required pattern")
			
			# Custom validator function
			if 'validator' in validation_rules:
				validator_func = validation_rules['validator']
				if callable(validator_func):
					try:
						if not validator_func(value):
							errors.append(f"{property_def.label} validation failed")
					except Exception as e:
						errors.append(f"Custom validation error: {e}")
			
		except Exception as e:
			errors.append(f"Custom validation error: {e}")
		
		return errors
	
	async def _update_dependent_properties(self, panel_state: PropertyPanelState, changed_property: str) -> None:
		"""Update properties that depend on the changed property."""
		try:
			for prop in panel_state.properties:
				if prop.depends_on and changed_property in prop.depends_on:
					# Evaluate condition
					await self._evaluate_property_condition(panel_state, prop)
			
		except Exception as e:
			logger.error(f"Failed to update dependent properties: {e}")
	
	async def _evaluate_property_condition(self, panel_state: PropertyPanelState, property_def: PropertyDefinition) -> None:
		"""Evaluate property visibility/enablement condition."""
		try:
			if not property_def.condition:
				return
			
			condition = property_def.condition
			condition_type = condition.get('type', 'visibility')
			expression = condition.get('expression')
			
			if not expression:
				return
			
			# Build evaluation context
			context = panel_state.values.copy()
			
			# Simple expression evaluation (could be enhanced with a proper expression engine)
			try:
				# Replace property references with actual values
				for prop_name, prop_value in context.items():
					expression = expression.replace(f'${prop_name}', str(prop_value))
				
				# Evaluate simple expressions
				result = eval(expression)  # Note: In production, use a safer expression evaluator
				
				if condition_type == 'visibility':
					property_def.visible = bool(result)
				elif condition_type == 'enabled':
					property_def.enabled = bool(result)
				
			except Exception as e:
				logger.warning(f"Failed to evaluate condition for property {property_def.name}: {e}")
			
		except Exception as e:
			logger.error(f"Failed to evaluate property condition: {e}")
	
	async def _register_builtin_widgets(self) -> None:
		"""Register built-in widget types."""
		try:
			builtin_widgets = {
				'input': {
					'type': 'text',
					'component': 'input',
					'props': ['placeholder', 'maxlength']
				},
				'textarea': {
					'type': 'textarea',
					'component': 'textarea',
					'props': ['placeholder', 'rows', 'cols']
				},
				'number': {
					'type': 'number',
					'component': 'input',
					'props': ['min', 'max', 'step']
				},
				'checkbox': {
					'type': 'checkbox',
					'component': 'input',
					'props': []
				},
				'select': {
					'type': 'select',
					'component': 'select',
					'props': ['options', 'multiple']
				},
				'code': {
					'type': 'code',
					'component': 'code-editor',
					'props': ['language', 'theme', 'height']
				},
				'json': {
					'type': 'json',
					'component': 'json-editor',
					'props': ['height']
				},
				'array': {
					'type': 'array',
					'component': 'array-editor',
					'props': ['item_type', 'min_items', 'max_items']
				},
				'file': {
					'type': 'file',
					'component': 'file-upload',
					'props': ['accept', 'multiple']
				},
				'color': {
					'type': 'color',
					'component': 'color-picker',
					'props': []
				},
				'date': {
					'type': 'date',
					'component': 'date-picker',
					'props': ['min_date', 'max_date']
				},
				'range': {
					'type': 'range',
					'component': 'range-slider',
					'props': ['min', 'max', 'step']
				}
			}
			
			for widget_type, config in builtin_widgets.items():
				self.widget_registry[widget_type] = config
			
			logger.info(f"Registered {len(builtin_widgets)} built-in widgets")
			
		except Exception as e:
			logger.error(f"Failed to register builtin widgets: {e}")
			raise