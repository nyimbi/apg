"""
APG Workflow Component Library

Comprehensive component library for workflow design with extensible components.

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

class ComponentPort(BaseModel):
	"""Represents a component input/output port."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	name: str = Field(..., description="Port name")
	type: str = Field(..., description="Data type (any, string, number, object, array)")
	required: bool = Field(default=False, description="Whether port is required")
	description: str = Field(default="", description="Port description")
	default_value: Any = Field(default=None, description="Default value")
	validation_schema: Optional[Dict[str, Any]] = Field(default=None, description="JSON schema for validation")

class ComponentProperty(BaseModel):
	"""Represents a configurable component property."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	name: str = Field(..., description="Property name")
	type: str = Field(..., description="Property type")
	label: str = Field(..., description="Display label")
	description: str = Field(default="", description="Property description")
	default_value: Any = Field(default=None, description="Default value")
	required: bool = Field(default=False, description="Whether property is required")
	
	# UI hints
	widget: str = Field(default="input", description="UI widget type")
	options: Optional[List[Dict[str, Any]]] = Field(default=None, description="Options for select widgets")
	validation: Optional[Dict[str, Any]] = Field(default=None, description="Validation rules")
	group: str = Field(default="general", description="Property group")

class ComponentDefinition(BaseModel):
	"""Complete definition of a workflow component."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Basic info
	id: str = Field(..., description="Unique component ID")
	name: str = Field(..., description="Component name")
	description: str = Field(..., description="Component description")
	category: str = Field(..., description="Component category")
	subcategory: str = Field(default="", description="Component subcategory")
	version: str = Field(default="1.0.0", description="Component version")
	
	# Visual
	icon: str = Field(default="fa-cog", description="Component icon")
	color: str = Field(default="#3498db", description="Component color")
	thumbnail: Optional[str] = Field(default=None, description="Component thumbnail URL")
	
	# Ports
	input_ports: List[ComponentPort] = Field(default_factory=list, description="Input ports")
	output_ports: List[ComponentPort] = Field(default_factory=list, description="Output ports")
	
	# Configuration
	properties: List[ComponentProperty] = Field(default_factory=list, description="Configurable properties")
	
	# Behavior
	execution_type: str = Field(default="sync", regex="^(sync|async|stream)$", description="Execution type")
	timeout: Optional[int] = Field(default=None, ge=1, description="Execution timeout in seconds")
	retries: int = Field(default=0, ge=0, le=10, description="Number of retries")
	
	# Metadata
	tags: List[str] = Field(default_factory=list, description="Component tags")
	documentation_url: Optional[str] = Field(default=None, description="Documentation URL")
	source_url: Optional[str] = Field(default=None, description="Source code URL")
	
	# Runtime
	runtime_requirements: Dict[str, Any] = Field(default_factory=dict, description="Runtime requirements")
	resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="Resource requirements")
	
	# Lifecycle
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	deprecated: bool = Field(default=False, description="Whether component is deprecated")

class ComponentCategory(BaseModel):
	"""Component category definition."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(..., description="Category ID")
	name: str = Field(..., description="Category name")
	description: str = Field(..., description="Category description")
	icon: str = Field(default="fa-folder", description="Category icon")
	color: str = Field(default="#95a5a6", description="Category color")
	order: int = Field(default=0, description="Display order")
	parent_id: Optional[str] = Field(default=None, description="Parent category ID")

class ComponentLibrary:
	"""
	Comprehensive component library for workflow design.
	
	Features:
	- Built-in components for common operations
	- Custom component support
	- Component validation and testing
	- Version management
	- Component marketplace integration
	"""
	
	def __init__(self, config):
		self.config = config
		self.components: Dict[str, ComponentDefinition] = {}
		self.categories: Dict[str, ComponentCategory] = {}
		self.custom_components: Dict[str, ComponentDefinition] = {}
		self.is_initialized = False
		
		logger.info("Component library initialized")
	
	async def initialize(self) -> None:
		"""Initialize the component library."""
		try:
			# Load built-in categories
			await self._load_builtin_categories()
			
			# Load built-in components
			await self._load_builtin_components()
			
			# Load custom components from database
			if self.config.lazy_loading:
				# Load metadata only, defer full loading
				await self._load_component_metadata()
			else:
				await self._load_custom_components()
			
			self.is_initialized = True
			logger.info(f"Component library initialized with {len(self.components)} components")
			
		except Exception as e:
			logger.error(f"Failed to initialize component library: {e}")
			raise
	
	async def shutdown(self) -> None:
		"""Shutdown the component library."""
		try:
			self.components.clear()
			self.categories.clear()
			self.custom_components.clear()
			self.is_initialized = False
			logger.info("Component library shutdown completed")
		except Exception as e:
			logger.error(f"Error during component library shutdown: {e}")
	
	async def get_component(self, component_id: str) -> Optional[ComponentDefinition]:
		"""Get component definition by ID."""
		try:
			# Check built-in components first
			if component_id in self.components:
				return self.components[component_id]
			
			# Check custom components
			if component_id in self.custom_components:
				return self.custom_components[component_id]
			
			# Try loading from database if lazy loading is enabled
			if self.config.lazy_loading:
				component = await self._load_component_from_database(component_id)
				if component:
					self.custom_components[component_id] = component
					return component
			
			return None
			
		except Exception as e:
			logger.error(f"Failed to get component {component_id}: {e}")
			return None
	
	async def get_components_by_category(self, category_id: str) -> List[ComponentDefinition]:
		"""Get all components in a category."""
		try:
			components = []
			
			# Built-in components
			for comp in self.components.values():
				if comp.category == category_id:
					components.append(comp)
			
			# Custom components
			for comp in self.custom_components.values():
				if comp.category == category_id:
					components.append(comp)
			
			return sorted(components, key=lambda x: x.name)
			
		except Exception as e:
			logger.error(f"Failed to get components by category: {e}")
			return []
	
	async def search_components(self, query: str, category: Optional[str] = None) -> List[ComponentDefinition]:
		"""Search components by name, description, or tags."""
		try:
			query_lower = query.lower()
			results = []
			
			all_components = {**self.components, **self.custom_components}
			
			for comp in all_components.values():
				# Skip if category filter doesn't match
				if category and comp.category != category:
					continue
				
				# Check name, description, and tags
				if (query_lower in comp.name.lower() or 
				    query_lower in comp.description.lower() or
				    any(query_lower in tag.lower() for tag in comp.tags)):
					results.append(comp)
			
			return sorted(results, key=lambda x: x.name)
			
		except Exception as e:
			logger.error(f"Failed to search components: {e}")
			return []
	
	async def get_categories(self) -> List[ComponentCategory]:
		"""Get all component categories."""
		try:
			return sorted(self.categories.values(), key=lambda x: (x.order, x.name))
		except Exception as e:
			logger.error(f"Failed to get categories: {e}")
			return []
	
	async def create_custom_component(self, definition: Dict[str, Any]) -> ComponentDefinition:
		"""Create a new custom component."""
		try:
			# Validate definition
			component = ComponentDefinition(**definition)
			
			# Generate ID if not provided
			if not component.id:
				component.id = f"custom_{str(uuid4())[:8]}"
			
			# Check for duplicates
			if component.id in self.components or component.id in self.custom_components:
				raise ValueError(f"Component with ID {component.id} already exists")
			
			# Save to database
			await self._save_component_to_database(component)
			
			# Add to library
			self.custom_components[component.id] = component
			
			logger.info(f"Created custom component {component.id}")
			return component
			
		except Exception as e:
			logger.error(f"Failed to create custom component: {e}")
			raise
	
	async def update_custom_component(self, component_id: str, definition: Dict[str, Any]) -> ComponentDefinition:
		"""Update a custom component."""
		try:
			if component_id not in self.custom_components:
				raise ValueError(f"Custom component {component_id} not found")
			
			# Update definition
			updated_definition = definition.copy()
			updated_definition['id'] = component_id
			updated_definition['updated_at'] = datetime.now(timezone.utc)
			
			component = ComponentDefinition(**updated_definition)
			
			# Save to database
			await self._update_component_in_database(component)
			
			# Update in library
			self.custom_components[component_id] = component
			
			logger.info(f"Updated custom component {component_id}")
			return component
			
		except Exception as e:
			logger.error(f"Failed to update custom component: {e}")
			raise
	
	async def delete_custom_component(self, component_id: str) -> None:
		"""Delete a custom component."""
		try:
			if component_id not in self.custom_components:
				raise ValueError(f"Custom component {component_id} not found")
			
			# Remove from database
			await self._delete_component_from_database(component_id)
			
			# Remove from library
			del self.custom_components[component_id]
			
			logger.info(f"Deleted custom component {component_id}")
			
		except Exception as e:
			logger.error(f"Failed to delete custom component: {e}")
			raise
	
	async def validate_component(self, definition: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate a component definition."""
		try:
			validation_result = {
				'valid': True,
				'errors': [],
				'warnings': []
			}
			
			# Basic validation using Pydantic
			try:
				ComponentDefinition(**definition)
			except Exception as e:
				validation_result['valid'] = False
				validation_result['errors'].append(f"Schema validation failed: {e}")
				return validation_result
			
			# Additional custom validations
			component = ComponentDefinition(**definition)
			
			# Check for required properties
			if not component.name.strip():
				validation_result['errors'].append("Component name is required")
			
			if not component.description.strip():
				validation_result['warnings'].append("Component description is recommended")
			
			# Validate ports
			for port in component.input_ports:
				if not port.name.strip():
					validation_result['errors'].append("Port names cannot be empty")
			
			for port in component.output_ports:
				if not port.name.strip():
					validation_result['errors'].append("Port names cannot be empty")
			
			# Validate properties
			for prop in component.properties:
				if not prop.name.strip():
					validation_result['errors'].append("Property names cannot be empty")
				if not prop.label.strip():
					validation_result['errors'].append("Property labels cannot be empty")
			
			# Check category exists
			if component.category not in self.categories:
				validation_result['warnings'].append(f"Category '{component.category}' does not exist")
			
			validation_result['valid'] = len(validation_result['errors']) == 0
			return validation_result
			
		except Exception as e:
			logger.error(f"Failed to validate component: {e}")
			return {
				'valid': False,
				'errors': [f"Validation error: {e}"],
				'warnings': []
			}
	
	async def get_component_usage_stats(self, component_id: str) -> Dict[str, Any]:
		"""Get usage statistics for a component."""
		try:
			from ..database import DatabaseManager
			
			db_manager = DatabaseManager()
			async with db_manager.get_session() as session:
				# Query usage from workflow definitions
				query = """
				SELECT 
					COUNT(*) as usage_count,
					COUNT(DISTINCT w.id) as workflow_count,
					MAX(w.updated_at) as last_used
				FROM cr_workflows w,
				     json_array_elements(w.definition->'nodes') as node
				WHERE node->>'component_type' = %s
				"""
				
				result = await session.execute(query, (component_id,))
				row = await result.fetchone()
				
				return {
					'component_id': component_id,
					'usage_count': row['usage_count'] or 0,
					'workflow_count': row['workflow_count'] or 0,
					'last_used': row['last_used'].isoformat() if row['last_used'] else None
				}
				
		except Exception as e:
			logger.error(f"Failed to get component usage stats: {e}")
			return {
				'component_id': component_id,
				'usage_count': 0,
				'workflow_count': 0,
				'last_used': None
			}
	
	# Private methods
	
	async def _load_builtin_categories(self) -> None:
		"""Load built-in component categories."""
		try:
			builtin_categories = [
				{
					'id': 'triggers',
					'name': 'Triggers',
					'description': 'Workflow trigger components',
					'icon': 'fa-play-circle',
					'color': '#e74c3c',
					'order': 1
				},
				{
					'id': 'data',
					'name': 'Data',
					'description': 'Data processing components',
					'icon': 'fa-database',
					'color': '#3498db',
					'order': 2
				},
				{
					'id': 'logic',
					'name': 'Logic',
					'description': 'Flow control and logic components',
					'icon': 'fa-code-branch',
					'color': '#f39c12',
					'order': 3
				},
				{
					'id': 'integrations',
					'name': 'Integrations',
					'description': 'External system integrations',
					'icon': 'fa-plug',
					'color': '#9b59b6',
					'order': 4
				},
				{
					'id': 'ai_ml',
					'name': 'AI/ML',
					'description': 'Artificial intelligence and machine learning',
					'icon': 'fa-brain',
					'color': '#1abc9c',
					'order': 5
				},
				{
					'id': 'utilities',
					'name': 'Utilities',
					'description': 'Utility and helper components',
					'icon': 'fa-tools',
					'color': '#95a5a6',
					'order': 6
				}
			]
			
			for cat_data in builtin_categories:
				category = ComponentCategory(**cat_data)
				self.categories[category.id] = category
			
		except Exception as e:
			logger.error(f"Failed to load builtin categories: {e}")
			raise
	
	async def _load_builtin_components(self) -> None:
		"""Load built-in components."""
		try:
			builtin_components = [
				# Triggers
				{
					'id': 'http_trigger',
					'name': 'HTTP Trigger',
					'description': 'Trigger workflow via HTTP request',
					'category': 'triggers',
					'icon': 'fa-globe',
					'color': '#e74c3c',
					'output_ports': [
						{'name': 'output', 'type': 'object', 'description': 'Request data'}
					],
					'properties': [
						{'name': 'method', 'type': 'string', 'label': 'HTTP Method', 'widget': 'select', 'options': [
							{'value': 'GET', 'label': 'GET'},
							{'value': 'POST', 'label': 'POST'},
							{'value': 'PUT', 'label': 'PUT'},
							{'value': 'DELETE', 'label': 'DELETE'}
						], 'default_value': 'POST'},
						{'name': 'path', 'type': 'string', 'label': 'Endpoint Path', 'required': True}
					],
					'tags': ['trigger', 'http', 'webhook']
				},
				{
					'id': 'schedule_trigger',
					'name': 'Schedule Trigger',
					'description': 'Trigger workflow on schedule',
					'category': 'triggers',
					'icon': 'fa-clock',
					'color': '#e74c3c',
					'output_ports': [
						{'name': 'output', 'type': 'object', 'description': 'Trigger event data'}
					],
					'properties': [
						{'name': 'cron', 'type': 'string', 'label': 'Cron Expression', 'required': True, 'description': 'Cron expression for scheduling'},
						{'name': 'timezone', 'type': 'string', 'label': 'Timezone', 'default_value': 'UTC'}
					],
					'tags': ['trigger', 'schedule', 'cron']
				},
				
				# Data Processing
				{
					'id': 'data_transform',
					'name': 'Data Transform',
					'description': 'Transform data using JavaScript',
					'category': 'data',
					'icon': 'fa-exchange-alt',
					'color': '#3498db',
					'input_ports': [
						{'name': 'input', 'type': 'any', 'description': 'Input data'}
					],
					'output_ports': [
						{'name': 'output', 'type': 'any', 'description': 'Transformed data'}
					],
					'properties': [
						{'name': 'script', 'type': 'string', 'label': 'Transform Script', 'widget': 'code', 'required': True, 'description': 'JavaScript transformation code'}
					],
					'tags': ['data', 'transform', 'javascript']
				},
				{
					'id': 'filter_data',
					'name': 'Filter Data',
					'description': 'Filter data based on conditions',
					'category': 'data',
					'icon': 'fa-filter',
					'color': '#3498db',
					'input_ports': [
						{'name': 'input', 'type': 'array', 'description': 'Input array'}
					],
					'output_ports': [
						{'name': 'output', 'type': 'array', 'description': 'Filtered array'}
					],
					'properties': [
						{'name': 'condition', 'type': 'string', 'label': 'Filter Condition', 'required': True, 'description': 'JavaScript filter expression'}
					],
					'tags': ['data', 'filter', 'array']
				},
				
				# Logic
				{
					'id': 'condition',
					'name': 'Condition',
					'description': 'Conditional branching',
					'category': 'logic',
					'icon': 'fa-code-branch',
					'color': '#f39c12',
					'input_ports': [
						{'name': 'input', 'type': 'any', 'description': 'Input data'}
					],
					'output_ports': [
						{'name': 'true', 'type': 'any', 'description': 'True branch output'},
						{'name': 'false', 'type': 'any', 'description': 'False branch output'}
					],
					'properties': [
						{'name': 'condition', 'type': 'string', 'label': 'Condition', 'required': True, 'description': 'JavaScript condition expression'}
					],
					'tags': ['logic', 'condition', 'branch']
				},
				{
					'id': 'loop',
					'name': 'Loop',
					'description': 'Iterate over data',
					'category': 'logic',
					'icon': 'fa-redo',
					'color': '#f39c12',
					'input_ports': [
						{'name': 'input', 'type': 'array', 'description': 'Array to iterate'}
					],
					'output_ports': [
						{'name': 'item', 'type': 'any', 'description': 'Current item'},
						{'name': 'index', 'type': 'number', 'description': 'Current index'},
						{'name': 'done', 'type': 'array', 'description': 'Completed items'}
					],
					'properties': [
						{'name': 'max_iterations', 'type': 'number', 'label': 'Max Iterations', 'default_value': 1000}
					],
					'tags': ['logic', 'loop', 'iteration']
				},
				
				# Integrations
				{
					'id': 'http_request',
					'name': 'HTTP Request',
					'description': 'Make HTTP requests',
					'category': 'integrations',
					'icon': 'fa-arrow-right',
					'color': '#9b59b6',
					'input_ports': [
						{'name': 'input', 'type': 'object', 'description': 'Request data'}
					],
					'output_ports': [
						{'name': 'response', 'type': 'object', 'description': 'Response data'},
						{'name': 'error', 'type': 'object', 'description': 'Error information'}
					],
					'properties': [
						{'name': 'url', 'type': 'string', 'label': 'URL', 'required': True},
						{'name': 'method', 'type': 'string', 'label': 'Method', 'widget': 'select', 'options': [
							{'value': 'GET', 'label': 'GET'},
							{'value': 'POST', 'label': 'POST'},
							{'value': 'PUT', 'label': 'PUT'},
							{'value': 'DELETE', 'label': 'DELETE'}
						], 'default_value': 'GET'},
						{'name': 'headers', 'type': 'object', 'label': 'Headers', 'widget': 'json'},
						{'name': 'timeout', 'type': 'number', 'label': 'Timeout (seconds)', 'default_value': 30}
					],
					'tags': ['integration', 'http', 'api']
				},
				{
					'id': 'database_query',
					'name': 'Database Query',
					'description': 'Execute database queries',
					'category': 'integrations',
					'icon': 'fa-database',
					'color': '#9b59b6',
					'input_ports': [
						{'name': 'params', 'type': 'object', 'description': 'Query parameters'}
					],
					'output_ports': [
						{'name': 'result', 'type': 'array', 'description': 'Query results'},
						{'name': 'error', 'type': 'object', 'description': 'Error information'}
					],
					'properties': [
						{'name': 'connection', 'type': 'string', 'label': 'Database Connection', 'required': True},
						{'name': 'query', 'type': 'string', 'label': 'SQL Query', 'widget': 'code', 'required': True}
					],
					'tags': ['integration', 'database', 'sql']
				}
			]
			
			for comp_data in builtin_components:
				component = ComponentDefinition(**comp_data)
				self.components[component.id] = component
			
		except Exception as e:
			logger.error(f"Failed to load builtin components: {e}")
			raise
	
	async def _load_component_metadata(self) -> None:
		"""Load component metadata for lazy loading."""
		try:
			from ..database import DatabaseManager
			
			db_manager = DatabaseManager()
			async with db_manager.get_session() as session:
				query = """
				SELECT id, name, category, description, icon, color, tags
				FROM cr_custom_components
				WHERE deprecated = false
				"""
				result = await session.execute(query)
				rows = await result.fetchall()
				
				for row in rows:
					# Create minimal component definition for metadata
					metadata = {
						'id': row['id'],
						'name': row['name'],
						'category': row['category'],
						'description': row['description'],
						'icon': row['icon'],
						'color': row['color'],
						'tags': row['tags'] or []
					}
					
					# Store metadata only
					self.custom_components[row['id']] = ComponentDefinition(**metadata)
				
				logger.info(f"Loaded metadata for {len(rows)} custom components")
			
		except Exception as e:
			logger.error(f"Failed to load component metadata: {e}")
	
	async def _load_custom_components(self) -> None:
		"""Load all custom components from database."""
		try:
			from ..database import DatabaseManager
			
			db_manager = DatabaseManager()
			async with db_manager.get_session() as session:
				query = "SELECT * FROM cr_custom_components WHERE deprecated = false"
				result = await session.execute(query)
				rows = await result.fetchall()
				
				for row in rows:
					component_data = {
						'id': row['id'],
						'name': row['name'],
						'description': row['description'],
						'category': row['category'],
						'subcategory': row['subcategory'] or '',
						'version': row['version'] or '1.0.0',
						'icon': row['icon'] or 'fa-cog',
						'color': row['color'] or '#3498db',
						'input_ports': row['input_ports'] or [],
						'output_ports': row['output_ports'] or [],
						'properties': row['properties'] or [],
						'execution_type': row['execution_type'] or 'sync',
						'tags': row['tags'] or [],
						'created_at': row['created_at'],
						'updated_at': row['updated_at'],
						'deprecated': row['deprecated']
					}
					
					component = ComponentDefinition(**component_data)
					self.custom_components[component.id] = component
				
				logger.info(f"Loaded {len(rows)} custom components")
			
		except Exception as e:
			logger.error(f"Failed to load custom components: {e}")
	
	async def _load_component_from_database(self, component_id: str) -> Optional[ComponentDefinition]:
		"""Load a specific component from database."""
		try:
			from ..database import DatabaseManager
			
			db_manager = DatabaseManager()
			async with db_manager.get_session() as session:
				query = "SELECT * FROM cr_custom_components WHERE id = %s"
				result = await session.execute(query, (component_id,))
				row = await result.fetchone()
				
				if not row:
					return None
				
				component_data = {
					'id': row['id'],
					'name': row['name'],
					'description': row['description'],
					'category': row['category'],
					'subcategory': row['subcategory'] or '',
					'version': row['version'] or '1.0.0',
					'icon': row['icon'] or 'fa-cog',
					'color': row['color'] or '#3498db',
					'input_ports': row['input_ports'] or [],
					'output_ports': row['output_ports'] or [],
					'properties': row['properties'] or [],
					'execution_type': row['execution_type'] or 'sync',
					'tags': row['tags'] or [],
					'created_at': row['created_at'],
					'updated_at': row['updated_at'],
					'deprecated': row['deprecated']
				}
				
				return ComponentDefinition(**component_data)
			
		except Exception as e:
			logger.error(f"Failed to load component from database: {e}")
			return None
	
	async def _save_component_to_database(self, component: ComponentDefinition) -> None:
		"""Save component to database."""
		try:
			from ..database import DatabaseManager
			
			db_manager = DatabaseManager()
			async with db_manager.get_session() as session:
				query = """
				INSERT INTO cr_custom_components (
					id, name, description, category, subcategory, version,
					icon, color, input_ports, output_ports, properties,
					execution_type, tags, created_at, updated_at, deprecated
				) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
				"""
				
				await session.execute(query, (
					component.id,
					component.name,
					component.description,
					component.category,
					component.subcategory,
					component.version,
					component.icon,
					component.color,
					json.dumps([port.model_dump() for port in component.input_ports]),
					json.dumps([port.model_dump() for port in component.output_ports]),
					json.dumps([prop.model_dump() for prop in component.properties]),
					component.execution_type,
					json.dumps(component.tags),
					component.created_at,
					component.updated_at,
					component.deprecated
				))
				
				await session.commit()
			
		except Exception as e:
			logger.error(f"Failed to save component to database: {e}")
			raise
	
	async def _update_component_in_database(self, component: ComponentDefinition) -> None:
		"""Update component in database."""
		try:
			from ..database import DatabaseManager
			
			db_manager = DatabaseManager()
			async with db_manager.get_session() as session:
				query = """
				UPDATE cr_custom_components SET
					name = %s, description = %s, category = %s, subcategory = %s,
					version = %s, icon = %s, color = %s, input_ports = %s,
					output_ports = %s, properties = %s, execution_type = %s,
					tags = %s, updated_at = %s, deprecated = %s
				WHERE id = %s
				"""
				
				await session.execute(query, (
					component.name,
					component.description,
					component.category,
					component.subcategory,
					component.version,
					component.icon,
					component.color,
					json.dumps([port.model_dump() for port in component.input_ports]),
					json.dumps([port.model_dump() for port in component.output_ports]),
					json.dumps([prop.model_dump() for prop in component.properties]),
					component.execution_type,
					json.dumps(component.tags),
					component.updated_at,
					component.deprecated,
					component.id
				))
				
				await session.commit()
			
		except Exception as e:
			logger.error(f"Failed to update component in database: {e}")
			raise
	
	async def _delete_component_from_database(self, component_id: str) -> None:
		"""Delete component from database."""
		try:
			from ..database import DatabaseManager
			
			db_manager = DatabaseManager()
			async with db_manager.get_session() as session:
				query = "DELETE FROM cr_custom_components WHERE id = %s"
				await session.execute(query, (component_id,))
				await session.commit()
			
		except Exception as e:
			logger.error(f"Failed to delete component from database: {e}")
			raise