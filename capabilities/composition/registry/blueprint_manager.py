"""
Blueprint Manager

Manages dynamic Flask blueprint registration and configuration for composed
APG applications. Handles blueprint discovery, conflict resolution, and
runtime management of Flask application structure.
"""

import os
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, validator
from flask import Flask, Blueprint
from flask_appbuilder import AppBuilder
from uuid_extensions import uuid7str

from .registry import CapabilityRegistry, CapabilityMetadata, SubCapabilityMetadata, get_registry

logger = logging.getLogger(__name__)

class BlueprintType(Enum):
	"""Types of blueprints."""
	CAPABILITY = "capability"  # Main capability blueprint
	SUBCAPABILITY = "subcapability"  # Sub-capability blueprint
	API = "api"  # API endpoints blueprint
	ADMIN = "admin"  # Administrative blueprint
	DASHBOARD = "dashboard"  # Dashboard blueprint
	STATIC = "static"  # Static resources blueprint

class BlueprintStatus(Enum):
	"""Blueprint registration status."""
	DISCOVERED = "discovered"
	REGISTERED = "registered"
	ACTIVE = "active"
	INACTIVE = "inactive"
	ERROR = "error"

class BlueprintConfiguration(BaseModel):
	"""Configuration for a blueprint."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	# Blueprint identity
	name: str = Field(..., description="Blueprint name")
	url_prefix: str = Field(..., description="URL prefix for blueprint")
	blueprint_type: BlueprintType = Field(..., description="Type of blueprint")
	
	# Registration details
	capability: str = Field(..., description="Parent capability")
	subcapability: Optional[str] = Field(default=None, description="Sub-capability if applicable")
	module_path: str = Field(..., description="Python module path")
	
	# Configuration
	template_folder: Optional[str] = Field(default=None, description="Template folder path")
	static_folder: Optional[str] = Field(default=None, description="Static folder path")
	static_url_path: Optional[str] = Field(default=None, description="Static URL path")
	
	# Blueprint options
	subdomain: Optional[str] = Field(default=None, description="Subdomain for blueprint")
	url_defaults: Optional[dict[str, Any]] = Field(default=None, description="Default URL values")
	root_path: Optional[str] = Field(default=None, description="Root path for blueprint")
	
	# Security and permissions
	required_permissions: list[str] = Field(default_factory=list, description="Required permissions")
	authentication_required: bool = Field(default=True, description="Whether authentication is required")
	
	# Menu integration
	menu_category: Optional[str] = Field(default=None, description="Menu category")
	menu_icon: Optional[str] = Field(default=None, description="Menu icon")
	menu_order: int = Field(default=100, description="Menu order")
	
	# Metadata
	priority: int = Field(default=100, description="Registration priority")
	enabled: bool = Field(default=True, description="Whether blueprint is enabled")
	tags: list[str] = Field(default_factory=list, description="Blueprint tags")

class RegisteredBlueprint(BaseModel):
	"""Information about a registered blueprint."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique blueprint ID")
	configuration: BlueprintConfiguration = Field(..., description="Blueprint configuration")
	blueprint: Optional[Any] = Field(default=None, description="Flask Blueprint instance", exclude=True)
	
	# Registration state
	status: BlueprintStatus = Field(default=BlueprintStatus.DISCOVERED, description="Registration status")
	registered_at: Optional[datetime] = Field(default=None, description="Registration timestamp")
	error_message: Optional[str] = Field(default=None, description="Error message if registration failed")
	
	# Runtime information
	endpoints: list[str] = Field(default_factory=list, description="Registered endpoints")
	url_rules: list[str] = Field(default_factory=list, description="URL rules")
	view_functions: list[str] = Field(default_factory=list, description="View functions")
	
	# Dependencies
	depends_on: list[str] = Field(default_factory=list, description="Blueprint dependencies")
	required_by: list[str] = Field(default_factory=list, description="Blueprints that require this one")

class BlueprintManager:
	"""
	Dynamic Flask blueprint manager for APG applications.
	
	Handles discovery, registration, and management of Flask blueprints
	from composed capabilities and sub-capabilities.
	"""
	
	def __init__(self, registry: Optional[CapabilityRegistry] = None):
		"""Initialize the blueprint manager."""
		self.registry = registry or get_registry()
		self.blueprints: Dict[str, RegisteredBlueprint] = {}
		self.flask_app: Optional[Flask] = None
		self.app_builder: Optional[AppBuilder] = None
		
		logger.info("BlueprintManager initialized")
	
	def discover_blueprints(self, capabilities: List[str]) -> Dict[str, BlueprintConfiguration]:
		"""
		Discover all blueprints for the given capabilities.
		
		Args:
			capabilities: List of capability codes
			
		Returns:
			Dictionary mapping blueprint names to configurations
		"""
		discovered_blueprints = {}
		
		# Ensure registry is populated
		if not self.registry.capabilities:
			self.registry.discover_all()
		
		for cap_code in capabilities:
			capability = self.registry.get_capability(cap_code)
			if not capability:
				logger.warning(f"Capability {cap_code} not found during blueprint discovery")
				continue
			
			# Discover capability-level blueprint
			cap_blueprint = self._discover_capability_blueprint(capability)
			if cap_blueprint:
				discovered_blueprints[cap_blueprint.name] = cap_blueprint
			
			# Discover sub-capability blueprints
			for subcap in capability.subcapabilities.values():
				subcap_blueprint = self._discover_subcapability_blueprint(capability, subcap)
				if subcap_blueprint:
					discovered_blueprints[subcap_blueprint.name] = subcap_blueprint
		
		logger.info(f"Discovered {len(discovered_blueprints)} blueprints")
		return discovered_blueprints
	
	def _discover_capability_blueprint(self, capability: CapabilityMetadata) -> Optional[BlueprintConfiguration]:
		"""Discover blueprint for a main capability."""
		try:
			# Try to import the capability module
			module = importlib.import_module(capability.module_path)
			
			# Look for blueprint creation functions
			blueprint_func = None
			if hasattr(module, 'create_capability_blueprint'):
				blueprint_func = 'create_capability_blueprint'
			elif hasattr(module, 'get_blueprint'):
				blueprint_func = 'get_blueprint'
			elif hasattr(module, 'blueprint'):
				# Check if it's a Blueprint instance
				blueprint_obj = getattr(module, 'blueprint')
				if isinstance(blueprint_obj, Blueprint):
					return self._create_blueprint_config_from_instance(
						blueprint_obj, capability, None
					)
			
			if blueprint_func:
				return BlueprintConfiguration(
					name=f"{capability.code.lower()}_main",
					url_prefix=f"/{capability.code.lower()}",
					blueprint_type=BlueprintType.CAPABILITY,
					capability=capability.code,
					module_path=capability.module_path,
					menu_category=capability.name,
					menu_icon="fa-cube"
				)
		
		except Exception as e:
			logger.debug(f"No capability blueprint found for {capability.code}: {e}")
		
		return None
	
	def _discover_subcapability_blueprint(self, 
										  capability: CapabilityMetadata,
										  subcapability: SubCapabilityMetadata) -> Optional[BlueprintConfiguration]:
		"""Discover blueprint for a sub-capability."""
		if not subcapability.has_blueprint:
			return None
		
		try:
			# Try to import the sub-capability blueprint module
			blueprint_module_path = f"{subcapability.module_path}.blueprint"
			module = importlib.import_module(blueprint_module_path)
			
			# Check for blueprint creation functions
			if hasattr(module, 'create_blueprint') or hasattr(module, 'get_blueprint'):
				return BlueprintConfiguration(
					name=f"{capability.code.lower()}_{subcapability.code}",
					url_prefix=f"/{capability.code.lower()}/{subcapability.code}",
					blueprint_type=BlueprintType.SUBCAPABILITY,
					capability=capability.code,
					subcapability=subcapability.code,
					module_path=blueprint_module_path,
					menu_category=capability.name,
					menu_icon="fa-cog"
				)
			
			# Check for direct blueprint instance
			elif hasattr(module, 'blueprint'):
				blueprint_obj = getattr(module, 'blueprint')
				if isinstance(blueprint_obj, Blueprint):
					return self._create_blueprint_config_from_instance(
						blueprint_obj, capability, subcapability
					)
		
		except Exception as e:
			logger.debug(f"No blueprint found for {capability.code}.{subcapability.code}: {e}")
		
		return None
	
	def _create_blueprint_config_from_instance(self, 
											   blueprint: Blueprint,
											   capability: CapabilityMetadata,
											   subcapability: Optional[SubCapabilityMetadata]) -> BlueprintConfiguration:
		"""Create blueprint configuration from a Blueprint instance."""
		subcap_name = subcapability.code if subcapability else "main"
		
		return BlueprintConfiguration(
			name=f"{capability.code.lower()}_{subcap_name}",
			url_prefix=blueprint.url_prefix or f"/{capability.code.lower()}",
			blueprint_type=BlueprintType.SUBCAPABILITY if subcapability else BlueprintType.CAPABILITY,
			capability=capability.code,
			subcapability=subcapability.code if subcapability else None,
			module_path=capability.module_path,
			template_folder=blueprint.template_folder,
			static_folder=blueprint.static_folder,
			static_url_path=blueprint.static_url_path,
			menu_category=capability.name
		)
	
	def register_blueprints(self, 
						   app: Flask,
						   app_builder: AppBuilder,
						   blueprint_configs: Dict[str, BlueprintConfiguration]) -> Dict[str, RegisteredBlueprint]:
		"""
		Register all blueprints with the Flask application.
		
		Args:
			app: Flask application instance
			app_builder: Flask-AppBuilder instance
			blueprint_configs: Blueprint configurations to register
			
		Returns:
			Dictionary of registered blueprints
		"""
		self.flask_app = app
		self.app_builder = app_builder
		
		# Sort blueprints by priority
		sorted_configs = sorted(
			blueprint_configs.items(),
			key=lambda x: x[1].priority
		)
		
		for blueprint_name, config in sorted_configs:
			try:
				registered_bp = self._register_single_blueprint(config)
				self.blueprints[blueprint_name] = registered_bp
				
				logger.info(f"Successfully registered blueprint: {blueprint_name}")
				
			except Exception as e:
				error_msg = f"Failed to register blueprint {blueprint_name}: {str(e)}"
				logger.error(error_msg)
				
				# Create error blueprint entry
				self.blueprints[blueprint_name] = RegisteredBlueprint(
					configuration=config,
					status=BlueprintStatus.ERROR,
					error_message=error_msg
				)
		
		logger.info(f"Registered {len([bp for bp in self.blueprints.values() if bp.status == BlueprintStatus.REGISTERED])} blueprints successfully")
		return self.blueprints
	
	def _register_single_blueprint(self, config: BlueprintConfiguration) -> RegisteredBlueprint:
		"""Register a single blueprint."""
		try:
			# Import the module
			module = importlib.import_module(config.module_path)
			blueprint = None
			
			# Try different methods to get/create the blueprint
			if hasattr(module, 'create_blueprint'):
				blueprint = module.create_blueprint()
			elif hasattr(module, 'get_blueprint'):
				blueprint = module.get_blueprint()
			elif hasattr(module, 'blueprint'):
				blueprint = getattr(module, 'blueprint')
			elif hasattr(module, 'create_capability_blueprint'):
				blueprint = module.create_capability_blueprint()
			
			if not blueprint or not isinstance(blueprint, Blueprint):
				raise ValueError("No valid Blueprint instance found")
			
			# Register the blueprint with Flask
			self.flask_app.register_blueprint(blueprint, url_prefix=config.url_prefix)
			
			# Register with Flask-AppBuilder if it has views
			if hasattr(module, 'register_capability_views'):
				try:
					subcaps = [config.subcapability] if config.subcapability else []
					module.register_capability_views(self.app_builder, subcaps)
				except Exception as e:
					logger.warning(f"Failed to register AppBuilder views for {config.name}: {e}")
			
			elif hasattr(module, 'init_capability'):
				try:
					subcaps = [config.subcapability] if config.subcapability else []
					module.init_capability(self.app_builder, subcaps)
				except Exception as e:
					logger.warning(f"Failed to initialize capability for {config.name}: {e}")
			
			# Extract blueprint information
			endpoints = []
			url_rules = []
			view_functions = []
			
			# Get blueprint endpoints
			for rule in self.flask_app.url_map.iter_rules():
				if rule.endpoint.startswith(blueprint.name + '.'):
					endpoints.append(rule.endpoint)
					url_rules.append(str(rule))
					
					view_func = self.flask_app.view_functions.get(rule.endpoint)
					if view_func:
						view_functions.append(view_func.__name__)
			
			# Create registered blueprint
			registered_bp = RegisteredBlueprint(
				configuration=config,
				blueprint=blueprint,
				status=BlueprintStatus.REGISTERED,
				registered_at=datetime.utcnow(),
				endpoints=endpoints,
				url_rules=url_rules,
				view_functions=view_functions
			)
			
			return registered_bp
			
		except Exception as e:
			raise Exception(f"Blueprint registration failed: {str(e)}")
	
	def get_blueprint_info(self, blueprint_name: str) -> Optional[Dict[str, Any]]:
		"""Get information about a registered blueprint."""
		blueprint = self.blueprints.get(blueprint_name)
		if not blueprint:
			return None
		
		return {
			'name': blueprint_name,
			'status': blueprint.status.value,
			'configuration': blueprint.configuration.dict(),
			'registered_at': blueprint.registered_at.isoformat() if blueprint.registered_at else None,
			'endpoints': blueprint.endpoints,
			'url_rules': blueprint.url_rules,
			'view_functions': blueprint.view_functions,
			'error_message': blueprint.error_message
		}
	
	def list_blueprints(self, status_filter: Optional[BlueprintStatus] = None) -> List[Dict[str, Any]]:
		"""List all registered blueprints with optional status filter."""
		blueprints = []
		
		for name, blueprint in self.blueprints.items():
			if status_filter is None or blueprint.status == status_filter:
				blueprints.append(self.get_blueprint_info(name))
		
		return blueprints
	
	def get_blueprint_by_capability(self, capability: str) -> List[Dict[str, Any]]:
		"""Get all blueprints for a specific capability."""
		return [
			self.get_blueprint_info(name)
			for name, blueprint in self.blueprints.items()
			if blueprint.configuration.capability == capability
		]
	
	def get_endpoints_by_capability(self, capability: str) -> List[str]:
		"""Get all endpoints for a specific capability."""
		endpoints = []
		
		for blueprint in self.blueprints.values():
			if blueprint.configuration.capability == capability:
				endpoints.extend(blueprint.endpoints)
		
		return endpoints
	
	def validate_blueprint_conflicts(self, configs: Dict[str, BlueprintConfiguration]) -> Dict[str, Any]:
		"""Validate blueprint configurations for conflicts."""
		validation_result = {
			"valid": True,
			"errors": [],
			"warnings": []
		}
		
		# Check for URL prefix conflicts
		url_prefixes = {}
		for name, config in configs.items():
			if config.url_prefix in url_prefixes:
				existing = url_prefixes[config.url_prefix]
				validation_result["errors"].append(
					f"URL prefix conflict: '{config.url_prefix}' used by both {name} and {existing}"
				)
				validation_result["valid"] = False
			else:
				url_prefixes[config.url_prefix] = name
		
		# Check for blueprint name conflicts
		blueprint_names = {}
		for name, config in configs.items():
			if config.name in blueprint_names:
				existing = blueprint_names[config.name]
				validation_result["errors"].append(
					f"Blueprint name conflict: '{config.name}' used by both {name} and {existing}"
				)
				validation_result["valid"] = False
			else:
				blueprint_names[config.name] = name
		
		# Check for subdomain conflicts
		subdomains = {}
		for name, config in configs.items():
			if config.subdomain:
				if config.subdomain in subdomains:
					existing = subdomains[config.subdomain]
					validation_result["warnings"].append(
						f"Subdomain '{config.subdomain}' used by both {name} and {existing}"
					)
				else:
					subdomains[config.subdomain] = name
		
		return validation_result
	
	def create_menu_structure(self) -> Dict[str, Any]:
		"""Create menu structure from registered blueprints."""
		menu_structure = {}
		
		for blueprint in self.blueprints.values():
			if blueprint.status != BlueprintStatus.REGISTERED:
				continue
			
			config = blueprint.configuration
			if not config.menu_category:
				continue
			
			category = config.menu_category
			if category not in menu_structure:
				menu_structure[category] = {
					"name": category,
					"icon": config.menu_icon or "fa-folder",
					"items": []
				}
			
			# Add menu item
			menu_item = {
				"name": config.name.replace('_', ' ').title(),
				"url": config.url_prefix or f"/{config.name}",
				"icon": config.menu_icon or "fa-cog",
				"order": config.menu_order,
				"capability": config.capability,
				"subcapability": config.subcapability
			}
			
			menu_structure[category]["items"].append(menu_item)
		
		# Sort menu items by order
		for category in menu_structure.values():
			category["items"].sort(key=lambda x: x["order"])
		
		return menu_structure
	
	def reload_blueprint(self, blueprint_name: str) -> bool:
		"""Reload a specific blueprint (useful for development)."""
		if blueprint_name not in self.blueprints:
			logger.error(f"Blueprint {blueprint_name} not found")
			return False
		
		try:
			registered_bp = self.blueprints[blueprint_name]
			config = registered_bp.configuration
			
			# Remove old blueprint
			if registered_bp.blueprint:
				# Flask doesn't provide a clean way to unregister blueprints
				# This is mainly for development use
				pass
			
			# Re-register blueprint
			new_bp = self._register_single_blueprint(config)
			self.blueprints[blueprint_name] = new_bp
			
			logger.info(f"Successfully reloaded blueprint: {blueprint_name}")
			return True
			
		except Exception as e:
			logger.error(f"Failed to reload blueprint {blueprint_name}: {e}")
			return False
	
	def get_blueprint_dependencies(self) -> Dict[str, List[str]]:
		"""Get dependency relationships between blueprints."""
		dependencies = {}
		
		for name, blueprint in self.blueprints.items():
			deps = []
			
			# Capability-level dependencies
			if blueprint.configuration.capability:
				capability = self.registry.get_capability(blueprint.configuration.capability)
				if capability:
					for dep in capability.dependencies:
						if dep.required:
							deps.append(dep.capability)
			
			dependencies[name] = deps
		
		return dependencies
	
	def check_blueprint_health(self) -> Dict[str, Any]:
		"""Check health status of all registered blueprints."""
		health_status = {
			"healthy": [],
			"unhealthy": [],
			"total_blueprints": len(self.blueprints),
			"registered_count": 0,
			"error_count": 0
		}
		
		for name, blueprint in self.blueprints.items():
			if blueprint.status == BlueprintStatus.REGISTERED:
				health_status["healthy"].append(name)
				health_status["registered_count"] += 1
			else:
				health_status["unhealthy"].append({
					"name": name,
					"status": blueprint.status.value,
					"error": blueprint.error_message
				})
				if blueprint.status == BlueprintStatus.ERROR:
					health_status["error_count"] += 1
		
		health_status["health_percentage"] = (
			health_status["registered_count"] / health_status["total_blueprints"] * 100
			if health_status["total_blueprints"] > 0 else 0
		)
		
		return health_status

# Global blueprint manager instance
_blueprint_manager_instance: Optional[BlueprintManager] = None

def get_blueprint_manager() -> BlueprintManager:
	"""Get the global blueprint manager instance."""
	global _blueprint_manager_instance
	if _blueprint_manager_instance is None:
		_blueprint_manager_instance = BlueprintManager()
	return _blueprint_manager_instance