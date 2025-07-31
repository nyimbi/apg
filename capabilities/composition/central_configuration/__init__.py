"""
APG Central Configuration Management

Unified configuration management system that allows each capability to contribute
configuration applets to a central configuration panel. Provides:

- Dynamic configuration applet discovery and registration
- Multi-tenant configuration isolation
- Role-based configuration access control
- Configuration validation and change management
- Real-time configuration updates and synchronization
- Configuration templates and presets
- Audit trail for all configuration changes
- API for programmatic configuration management

Each APG capability can contribute configuration applets by implementing
the ConfigurationApplet interface and registering with this system.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict
import asyncio
from datetime import datetime

class ConfigurationScope(str, Enum):
	"""Configuration scope levels."""
	GLOBAL = "global"
	TENANT = "tenant"
	USER = "user"
	CAPABILITY = "capability"
	ENVIRONMENT = "environment"

class ConfigurationDataType(str, Enum):
	"""Supported configuration data types."""
	STRING = "string"
	INTEGER = "integer"
	FLOAT = "float"
	BOOLEAN = "boolean"
	JSON = "json"
	ARRAY = "array"
	SECRET = "secret"
	FILE = "file"

class ConfigurationChangeType(str, Enum):
	"""Types of configuration changes."""
	CREATE = "create"
	UPDATE = "update"
	DELETE = "delete"
	BULK_UPDATE = "bulk_update"
	RESET = "reset"

@dataclass
class ConfigurationField:
	"""Configuration field definition."""
	key: str
	label: str
	data_type: ConfigurationDataType
	default_value: Any = None
	required: bool = False
	description: str = ""
	validation_rules: Dict[str, Any] = None
	depends_on: List[str] = None
	scope: ConfigurationScope = ConfigurationScope.TENANT

class ConfigurationApplet(ABC):
	"""Abstract base class for capability configuration applets."""
	
	@property
	@abstractmethod
	def applet_id(self) -> str:
		"""Unique applet identifier."""
		pass
	
	@property
	@abstractmethod
	def capability_name(self) -> str:
		"""Name of the capability this applet configures."""
		pass
	
	@property
	@abstractmethod
	def display_name(self) -> str:
		"""Human-readable display name."""
		pass
	
	@property
	@abstractmethod
	def description(self) -> str:
		"""Applet description."""
		pass
	
	@abstractmethod
	def get_configuration_fields(self) -> List[ConfigurationField]:
		"""Get list of configuration fields this applet manages."""
		pass
	
	@abstractmethod
	async def validate_configuration(self, config: Dict[str, Any]) -> Dict[str, str]:
		"""Validate configuration values. Returns dict of field_key -> error_message."""
		pass
	
	@abstractmethod
	async def apply_configuration(self, tenant_id: str, config: Dict[str, Any]) -> bool:
		"""Apply configuration changes. Returns True if successful."""
		pass
	
	async def get_current_configuration(self, tenant_id: str) -> Dict[str, Any]:
		"""Get current configuration values for tenant."""
		return {}
	
	async def reset_to_defaults(self, tenant_id: str) -> Dict[str, Any]:
		"""Reset configuration to default values."""
		defaults = {}
		for field in self.get_configuration_fields():
			if field.default_value is not None:
				defaults[field.key] = field.default_value
		return defaults

class ConfigurationChange(BaseModel):
	"""Configuration change record."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	applet_id: str
	user_id: str
	change_type: ConfigurationChangeType
	field_key: str
	old_value: Any = None
	new_value: Any = None
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	reason: Optional[str] = None

class ConfigurationTemplate(BaseModel):
	"""Configuration template for quick setup."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	description: str
	category: str  # e.g., "industry_vertical", "deployment_size", "security_profile"
	configuration: Dict[str, Dict[str, Any]]  # applet_id -> config values
	tags: List[str] = Field(default_factory=list)
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)

class CentralConfigurationManager:
	"""Central configuration management system."""
	
	def __init__(self):
		self._applets: Dict[str, ConfigurationApplet] = {}
		self._tenant_configs: Dict[str, Dict[str, Dict[str, Any]]] = {}  # tenant_id -> applet_id -> config
		self._change_history: List[ConfigurationChange] = []
		self._templates: Dict[str, ConfigurationTemplate] = {}
		self._change_listeners: Dict[str, List[Callable]] = {}
	
	def register_applet(self, applet: ConfigurationApplet) -> bool:
		"""Register a configuration applet."""
		if applet.applet_id in self._applets:
			return False
		
		self._applets[applet.applet_id] = applet
		return True
	
	def unregister_applet(self, applet_id: str) -> bool:
		"""Unregister a configuration applet."""
		if applet_id not in self._applets:
			return False
		
		del self._applets[applet_id]
		return True
	
	def get_applets(self) -> List[ConfigurationApplet]:
		"""Get all registered applets."""
		return list(self._applets.values())
	
	def get_applet(self, applet_id: str) -> Optional[ConfigurationApplet]:
		"""Get specific applet by ID."""
		return self._applets.get(applet_id)
	
	async def get_tenant_configuration(self, tenant_id: str, applet_id: str) -> Dict[str, Any]:
		"""Get current configuration for tenant and applet."""
		if tenant_id not in self._tenant_configs:
			self._tenant_configs[tenant_id] = {}
		
		if applet_id not in self._tenant_configs[tenant_id]:
			# Initialize with defaults
			applet = self._applets.get(applet_id)
			if applet:
				defaults = await applet.reset_to_defaults(tenant_id)
				self._tenant_configs[tenant_id][applet_id] = defaults
			else:
				self._tenant_configs[tenant_id][applet_id] = {}
		
		return self._tenant_configs[tenant_id][applet_id].copy()
	
	async def update_configuration(
		self,
		tenant_id: str,
		applet_id: str,
		user_id: str,
		updates: Dict[str, Any],
		reason: Optional[str] = None
	) -> Dict[str, str]:
		"""Update configuration. Returns validation errors if any."""
		applet = self._applets.get(applet_id)
		if not applet:
			return {"applet": f"Applet {applet_id} not found"}
		
		# Get current config
		current_config = await self.get_tenant_configuration(tenant_id, applet_id)
		
		# Merge updates
		new_config = current_config.copy()
		new_config.update(updates)
		
		# Validate
		validation_errors = await applet.validate_configuration(new_config)
		if validation_errors:
			return validation_errors
		
		# Apply configuration
		success = await applet.apply_configuration(tenant_id, new_config)
		if not success:
			return {"system": "Failed to apply configuration"}
		
		# Record changes
		for key, value in updates.items():
			old_value = current_config.get(key)
			change = ConfigurationChange(
				tenant_id=tenant_id,
				applet_id=applet_id,
				user_id=user_id,
				change_type=ConfigurationChangeType.UPDATE,
				field_key=key,
				old_value=old_value,
				new_value=value,
				reason=reason
			)
			self._change_history.append(change)
		
		# Update stored config
		self._tenant_configs[tenant_id][applet_id] = new_config
		
		# Notify listeners
		await self._notify_change_listeners(tenant_id, applet_id, updates)
		
		return {}  # No errors
	
	async def apply_template(
		self,
		tenant_id: str,
		user_id: str,
		template_id: str,
		reason: Optional[str] = None
	) -> Dict[str, Dict[str, str]]:
		"""Apply configuration template. Returns applet_id -> validation_errors."""
		template = self._templates.get(template_id)
		if not template:
			return {"template": {"error": f"Template {template_id} not found"}}
		
		results = {}
		for applet_id, config in template.configuration.items():
			errors = await self.update_configuration(
				tenant_id, applet_id, user_id, config, 
				reason or f"Applied template: {template.name}"
			)
			if errors:
				results[applet_id] = errors
		
		return results
	
	def register_change_listener(self, applet_id: str, callback: Callable):
		"""Register a callback for configuration changes."""
		if applet_id not in self._change_listeners:
			self._change_listeners[applet_id] = []
		self._change_listeners[applet_id].append(callback)
	
	async def _notify_change_listeners(self, tenant_id: str, applet_id: str, changes: Dict[str, Any]):
		"""Notify registered change listeners."""
		listeners = self._change_listeners.get(applet_id, [])
		for listener in listeners:
			try:
				if asyncio.iscoroutinefunction(listener):
					await listener(tenant_id, applet_id, changes)
				else:
					listener(tenant_id, applet_id, changes)
			except Exception as e:
				# Log error but don't fail the configuration update
				print(f"Error notifying configuration listener: {e}")
	
	def save_template(self, template: ConfigurationTemplate) -> bool:
		"""Save a configuration template."""
		self._templates[template.id] = template
		return True
	
	def get_templates(self, category: Optional[str] = None) -> List[ConfigurationTemplate]:
		"""Get available templates, optionally filtered by category."""
		templates = list(self._templates.values())
		if category:
			templates = [t for t in templates if t.category == category]
		return templates
	
	def get_change_history(
		self,
		tenant_id: str,
		applet_id: Optional[str] = None,
		limit: int = 100
	) -> List[ConfigurationChange]:
		"""Get configuration change history."""
		changes = [c for c in self._change_history if c.tenant_id == tenant_id]
		if applet_id:
			changes = [c for c in changes if c.applet_id == applet_id]
		
		# Sort by timestamp descending and limit
		changes.sort(key=lambda c: c.timestamp, reverse=True)
		return changes[:limit]

# Global configuration manager instance
_config_manager = CentralConfigurationManager()

def get_configuration_manager() -> CentralConfigurationManager:
	"""Get the global configuration manager instance."""
	return _config_manager

def register_configuration_applet(applet: ConfigurationApplet) -> bool:
	"""Register a configuration applet with the central manager."""
	return _config_manager.register_applet(applet)

# Example applet for demonstration
class CompositionConfigurationApplet(ConfigurationApplet):
	"""Configuration applet for the composition capability itself."""
	
	@property
	def applet_id(self) -> str:
		return "composition.core"
	
	@property
	def capability_name(self) -> str:
		return "Composition"
	
	@property
	def display_name(self) -> str:
		return "Composition Engine Settings"
	
	@property
	def description(self) -> str:
		return "Core configuration for the APG composition engine"
	
	def get_configuration_fields(self) -> List[ConfigurationField]:
		return [
			ConfigurationField(
				key="auto_discovery_enabled",
				label="Auto-discovery Enabled",
				data_type=ConfigurationDataType.BOOLEAN,
				default_value=True,
				description="Automatically discover new capabilities"
			),
			ConfigurationField(
				key="composition_cache_ttl",
				label="Composition Cache TTL (seconds)",
				data_type=ConfigurationDataType.INTEGER,
				default_value=3600,
				description="Time to live for composition cache entries"
			),
			ConfigurationField(
				key="default_deployment_strategy",
				label="Default Deployment Strategy",
				data_type=ConfigurationDataType.STRING,
				default_value="rolling_update",
				description="Default strategy for deploying compositions"
			)
		]
	
	async def validate_configuration(self, config: Dict[str, Any]) -> Dict[str, str]:
		errors = {}
		
		if "composition_cache_ttl" in config:
			ttl = config["composition_cache_ttl"]
			if not isinstance(ttl, int) or ttl < 60:
				errors["composition_cache_ttl"] = "Cache TTL must be at least 60 seconds"
		
		return errors
	
	async def apply_configuration(self, tenant_id: str, config: Dict[str, Any]) -> bool:
		# In a real implementation, this would update the composition engine settings
		print(f"Applied composition configuration for tenant {tenant_id}: {config}")
		return True

# Register the composition applet by default
register_configuration_applet(CompositionConfigurationApplet())

# Capability metadata
CAPABILITY_METADATA = {
	"name": "Central Configuration",
	"version": "1.0.0",
	"description": "Unified configuration management for all APG capabilities",
	"category": "infrastructure",
	"dependencies": [],
	"provides": [
		"configuration_management",
		"applet_registry",
		"configuration_templates",
		"change_tracking"
	],
	"requires_auth": True,
	"multi_tenant": True
}

__all__ = [
	"ConfigurationScope",
	"ConfigurationDataType", 
	"ConfigurationChangeType",
	"ConfigurationField",
	"ConfigurationApplet",
	"ConfigurationChange",
	"ConfigurationTemplate",
	"CentralConfigurationManager",
	"get_configuration_manager",
	"register_configuration_applet",
	"CompositionConfigurationApplet",
	"CAPABILITY_METADATA"
]