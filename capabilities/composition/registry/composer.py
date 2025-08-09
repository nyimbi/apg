"""
Composition Engine

Main composition engine for creating custom APG applications by selecting and
combining specific capabilities and sub-capabilities. Handles validation,
dependency resolution, Flask app generation, and configuration management.
"""

import os
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict, validator
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, JSON, Index
from sqlalchemy.orm import relationship
from flask import Flask
from flask_appbuilder import AppBuilder

from .registry import CapabilityRegistry, CapabilityMetadata, SubCapabilityMetadata, get_registry
from .validator import DependencyValidator, ValidationResult, get_validator

logger = logging.getLogger(__name__)

class CompositionConfig(BaseModel):
	"""Configuration for application composition."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	# Basic configuration
	tenant_id: str = Field(..., description="Tenant identifier")
	application_name: str = Field(..., description="Application name")
	application_version: str = Field(default="1.0.0", description="Application version")
	
	# Capability selection
	capabilities: list[str] = Field(..., description="List of capability codes to include")
	subcapabilities: dict[str, list[str]] = Field(default_factory=dict, description="Sub-capabilities per capability")
	
	# Industry and templates
	industry_template: Optional[str] = Field(default=None, description="Industry template to apply")
	base_template: Optional[str] = Field(default=None, description="Base template to start from")
	
	# Flask configuration
	flask_config: dict[str, Any] = Field(default_factory=dict, description="Flask app configuration")
	database_url: Optional[str] = Field(default=None, description="Database connection URL")
	secret_key: Optional[str] = Field(default=None, description="Flask secret key")
	
	# Feature flags
	enable_multi_tenancy: bool = Field(default=True, description="Enable multi-tenant support")
	enable_audit_logging: bool = Field(default=True, description="Enable audit logging")
	enable_api_docs: bool = Field(default=True, description="Enable API documentation")
	enable_security: bool = Field(default=True, description="Enable security features")
	
	# Custom overrides
	custom_config: dict[str, Any] = Field(default_factory=dict, description="Custom configuration overrides")
	excluded_features: list[str] = Field(default_factory=list, description="Features to exclude")
	
	# Development settings
	debug_mode: bool = Field(default=False, description="Enable debug mode")
	testing_mode: bool = Field(default=False, description="Enable testing mode")

class CompositionContext(BaseModel):
	"""Context information for composition process."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	composition_id: str = Field(default_factory=uuid7str, description="Unique composition ID")
	started_at: datetime = Field(default_factory=datetime.utcnow, description="Composition start time")
	config: CompositionConfig = Field(..., description="Composition configuration")
	
	# Registry and validation
	registry: Optional[Any] = Field(default=None, description="Capability registry", exclude=True)
	validation_result: Optional[ValidationResult] = Field(default=None, description="Validation result")
	
	# Resolved metadata
	resolved_capabilities: dict[str, CapabilityMetadata] = Field(default_factory=dict, description="Resolved capabilities")
	resolved_subcapabilities: dict[str, SubCapabilityMetadata] = Field(default_factory=dict, description="Resolved sub-capabilities")
	dependency_order: list[str] = Field(default_factory=list, description="Dependency resolution order")
	
	# Generated components
	flask_app: Optional[Any] = Field(default=None, description="Generated Flask app", exclude=True)
	app_builder: Optional[Any] = Field(default=None, description="AppBuilder instance", exclude=True)
	database_models: list[str] = Field(default_factory=list, description="Database models to create")
	
	# Composition results
	warnings: list[str] = Field(default_factory=list, description="Composition warnings")
	errors: list[str] = Field(default_factory=list, description="Composition errors")
	metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class CompositionResult(BaseModel):
	"""Result of application composition."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	success: bool = Field(..., description="Whether composition succeeded")
	composition_id: str = Field(..., description="Unique composition ID")
	config: CompositionConfig = Field(..., description="Composition configuration")
	
	# Generated application
	flask_app: Optional[Any] = Field(default=None, description="Generated Flask app", exclude=True)
	app_builder: Optional[Any] = Field(default=None, description="AppBuilder instance", exclude=True)
	
	# Metadata
	capabilities_included: list[str] = Field(default_factory=list, description="Capabilities included")
	subcapabilities_included: list[str] = Field(default_factory=list, description="Sub-capabilities included")
	database_models: list[str] = Field(default_factory=list, description="Database models")
	api_endpoints: list[str] = Field(default_factory=list, description="API endpoints")
	
	# Results
	warnings: list[str] = Field(default_factory=list, description="Composition warnings")
	errors: list[str] = Field(default_factory=list, description="Composition errors")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	
	# Performance metrics
	composition_time_ms: int = Field(default=0, description="Composition time in milliseconds")
	total_files_processed: int = Field(default=0, description="Total files processed")
	total_models_created: int = Field(default=0, description="Total models created")

class CompositionEngine:
	"""
	Main composition engine for creating APG applications.
	
	Orchestrates the entire process of building a custom ERP application
	from selected capabilities and sub-capabilities.
	"""
	
	def __init__(self, registry: Optional[CapabilityRegistry] = None):
		"""Initialize the composition engine."""
		self.registry = registry or get_registry()
		self.validator = get_validator()
		
		logger.info("CompositionEngine initialized")
	
	def compose(self, 
			   tenant_id: str,
			   capabilities: List[str],
			   industry_template: Optional[str] = None,
			   custom_config: Optional[Dict[str, Any]] = None) -> CompositionResult:
		"""
		Main composition method.
		
		Args:
			tenant_id: Unique tenant identifier
			capabilities: List of capability codes to include
			industry_template: Optional industry template to apply
			custom_config: Custom configuration overrides
			
		Returns:
			CompositionResult with generated application and metadata
		"""
		start_time = datetime.utcnow()
		
		try:
			# Create composition configuration
			config = CompositionConfig(
				tenant_id=tenant_id,
				application_name=f"APG_App_{tenant_id}",
				capabilities=capabilities,
				industry_template=industry_template,
				custom_config=custom_config or {}
			)
			
			# Create composition context
			context = CompositionContext(config=config, registry=self.registry)
			
			logger.info(f"Starting composition for tenant {tenant_id} with capabilities: {capabilities}")
			
			# Step 1: Validate composition
			self._validate_composition(context)
			if context.errors:
				return self._create_error_result(context, start_time)
			
			# Step 2: Resolve dependencies and capabilities
			self._resolve_dependencies(context)
			if context.errors:
				return self._create_error_result(context, start_time)
			
			# Step 3: Apply industry template if specified
			if industry_template:
				self._apply_industry_template(context)
			
			# Step 4: Create Flask application
			self._create_flask_app(context)
			if context.errors:
				return self._create_error_result(context, start_time)
			
			# Step 5: Initialize capabilities
			self._initialize_capabilities(context)
			if context.errors:
				return self._create_error_result(context, start_time)
			
			# Step 6: Finalize application
			self._finalize_application(context)
			
			# Create successful result
			end_time = datetime.utcnow()
			composition_time = int((end_time - start_time).total_seconds() * 1000)
			
			result = CompositionResult(
				success=True,
				composition_id=context.composition_id,
				config=config,
				flask_app=context.flask_app,
				app_builder=context.app_builder,
				capabilities_included=list(context.resolved_capabilities.keys()),
				subcapabilities_included=list(context.resolved_subcapabilities.keys()),
				database_models=context.database_models,
				warnings=context.warnings,
				errors=context.errors,
				composition_time_ms=composition_time
			)
			
			logger.info(f"Composition completed successfully in {composition_time}ms")
			return result
			
		except Exception as e:
			logger.error(f"Composition failed: {e}")
			context.errors.append(f"Composition failed: {str(e)}")
			return self._create_error_result(context, start_time)
	
	def _validate_composition(self, context: CompositionContext) -> None:
		"""Validate the composition configuration."""
		try:
			# Validate with dependency validator
			validation_result = self.validator.validate_composition(context.config.capabilities)
			context.validation_result = validation_result
			
			if not validation_result.valid:
				context.errors.extend(validation_result.errors)
				return
			
			context.warnings.extend(validation_result.warnings)
			
			# Check that all capabilities exist
			for cap_code in context.config.capabilities:
				capability = self.registry.get_capability(cap_code)
				if not capability:
					context.errors.append(f"Capability '{cap_code}' not found in registry")
			
		except Exception as e:
			context.errors.append(f"Validation failed: {str(e)}")
	
	def _resolve_dependencies(self, context: CompositionContext) -> None:
		"""Resolve all dependencies and build dependency order."""
		try:
			# Get all requested capabilities
			all_capabilities = set(context.config.capabilities)
			
			# Add dependencies recursively
			to_process = list(context.config.capabilities)
			processed = set()
			
			while to_process:
				cap_code = to_process.pop(0)
				if cap_code in processed:
					continue
				
				processed.add(cap_code)
				capability = self.registry.get_capability(cap_code)
				
				if capability:
					context.resolved_capabilities[cap_code] = capability
					
					# Add dependencies
					for dep in capability.dependencies:
						if dep.required and dep.capability not in all_capabilities:
							all_capabilities.add(dep.capability)
							to_process.append(dep.capability)
			
			# Resolve sub-capabilities
			for cap_code, capability in context.resolved_capabilities.items():
				# Get requested sub-capabilities for this capability
				requested_subcaps = context.config.subcapabilities.get(cap_code, [])
				
				if not requested_subcaps:
					# If no specific sub-capabilities requested, include all available
					requested_subcaps = list(capability.subcapabilities.keys())
				
				for subcap_code in requested_subcaps:
					if subcap_code in capability.subcapabilities:
						subcap_key = f"{cap_code}.{subcap_code}"
						context.resolved_subcapabilities[subcap_key] = capability.subcapabilities[subcap_code]
			
			# Build dependency order using topological sort
			context.dependency_order = self._topological_sort_capabilities(context.resolved_capabilities)
			
		except Exception as e:
			context.errors.append(f"Dependency resolution failed: {str(e)}")
	
	def _topological_sort_capabilities(self, capabilities: Dict[str, CapabilityMetadata]) -> List[str]:
		"""Sort capabilities by dependency order using topological sort."""
		# Build dependency graph
		graph = {}
		in_degree = {}
		
		for cap_code in capabilities:
			graph[cap_code] = []
			in_degree[cap_code] = 0
		
		for cap_code, capability in capabilities.items():
			for dep in capability.dependencies:
				if dep.capability in capabilities and dep.required:
					graph[dep.capability].append(cap_code)
					in_degree[cap_code] += 1
		
		# Topological sort
		queue = [cap for cap, degree in in_degree.items() if degree == 0]
		result = []
		
		while queue:
			cap = queue.pop(0)
			result.append(cap)
			
			for neighbor in graph[cap]:
				in_degree[neighbor] -= 1
				if in_degree[neighbor] == 0:
					queue.append(neighbor)
		
		# Check for circular dependencies
		if len(result) != len(capabilities):
			remaining = set(capabilities.keys()) - set(result)
			logger.warning(f"Circular dependencies detected among: {remaining}")
			result.extend(remaining)  # Add remaining in arbitrary order
		
		return result
	
	def _apply_industry_template(self, context: CompositionContext) -> None:
		"""Apply industry-specific template configurations."""
		try:
			from .templates import get_template_manager
			
			template_manager = get_template_manager()
			template = template_manager.get_template(context.config.industry_template)
			
			if template:
				# Apply template configurations
				if template.default_capabilities:
					# Add any missing default capabilities
					for cap in template.default_capabilities:
						if cap not in context.config.capabilities:
							context.config.capabilities.append(cap)
							context.warnings.append(f"Added default capability '{cap}' from industry template")
				
				# Apply template configuration overrides
				if template.configuration_overrides:
					context.config.custom_config.update(template.configuration_overrides)
				
				logger.info(f"Applied industry template: {context.config.industry_template}")
			else:
				context.warnings.append(f"Industry template '{context.config.industry_template}' not found")
				
		except Exception as e:
			context.warnings.append(f"Failed to apply industry template: {str(e)}")
	
	def _create_flask_app(self, context: CompositionContext) -> None:
		"""Create and configure the Flask application."""
		try:
			# Create Flask app
			app = Flask(context.config.application_name)
			
			# Basic Flask configuration
			app.config.update({
				'SECRET_KEY': context.config.secret_key or os.urandom(24).hex(),
				'SQLALCHEMY_DATABASE_URI': context.config.database_url or 'sqlite:///apg_app.db',
				'SQLALCHEMY_TRACK_MODIFICATIONS': False,
				'DEBUG': context.config.debug_mode,
				'TESTING': context.config.testing_mode,
			})
			
			# Apply custom Flask configuration
			if context.config.flask_config:
				app.config.update(context.config.flask_config)
			
			# Apply custom configuration overrides
			if context.config.custom_config:
				app.config.update(context.config.custom_config)
			
			# Initialize Flask-AppBuilder
			appbuilder = AppBuilder(app)
			
			context.flask_app = app
			context.app_builder = appbuilder
			
			logger.info("Flask application created successfully")
			
		except Exception as e:
			context.errors.append(f"Failed to create Flask app: {str(e)}")
	
	def _initialize_capabilities(self, context: CompositionContext) -> None:
		"""Initialize all capabilities in dependency order."""
		try:
			for cap_code in context.dependency_order:
				capability = context.resolved_capabilities[cap_code]
				
				try:
					self._initialize_capability(context, capability)
				except Exception as e:
					error_msg = f"Failed to initialize capability '{cap_code}': {str(e)}"
					context.errors.append(error_msg)
					logger.error(error_msg)
			
		except Exception as e:
			context.errors.append(f"Capability initialization failed: {str(e)}")
	
	def _initialize_capability(self, context: CompositionContext, capability: CapabilityMetadata) -> None:
		"""Initialize a single capability."""
		try:
			# Import the capability module
			module = importlib.import_module(capability.module_path)
			
			# Check for init_capability function
			if hasattr(module, 'init_capability'):
				# Get sub-capabilities for this capability
				subcaps = []
				for subcap_key, subcap in context.resolved_subcapabilities.items():
					if subcap.parent_capability == capability.code:
						subcaps.append(subcap.code)
				
				# Initialize the capability
				module.init_capability(context.app_builder, subcaps)
				logger.info(f"Initialized capability: {capability.code}")
			
			# Handle blueprint registration if available
			elif hasattr(module, 'register_capability_views'):
				subcaps = []
				for subcap_key, subcap in context.resolved_subcapabilities.items():
					if subcap.parent_capability == capability.code:
						subcaps.append(subcap.code)
				
				module.register_capability_views(context.app_builder, subcaps)
				logger.info(f"Registered capability views: {capability.code}")
			
			else:
				context.warnings.append(f"No initialization method found for capability: {capability.code}")
			
		except ImportError as e:
			context.warnings.append(f"Could not import capability {capability.code}: {str(e)}")
		except Exception as e:
			raise Exception(f"Error initializing capability {capability.code}: {str(e)}")
	
	def _finalize_application(self, context: CompositionContext) -> None:
		"""Finalize the application setup."""
		try:
			# Initialize database if needed
			if context.flask_app and context.app_builder:
				# Create database tables
				with context.flask_app.app_context():
					context.app_builder.get_app.db.create_all()
				
				logger.info("Database tables created")
			
			# Collect database models
			for capability in context.resolved_capabilities.values():
				for subcap in capability.subcapabilities.values():
					context.database_models.extend(subcap.database_tables)
			
			# Additional finalization steps can be added here
			
		except Exception as e:
			context.warnings.append(f"Application finalization warning: {str(e)}")
	
	def _create_error_result(self, context: CompositionContext, start_time: datetime) -> CompositionResult:
		"""Create a result object for failed composition."""
		end_time = datetime.utcnow()
		composition_time = int((end_time - start_time).total_seconds() * 1000)
		
		return CompositionResult(
			success=False,
			composition_id=context.composition_id,
			config=context.config,
			errors=context.errors,
			warnings=context.warnings,
			composition_time_ms=composition_time
		)
	
	def get_composition_info(self, capabilities: List[str]) -> Dict[str, Any]:
		"""Get information about what would be included in a composition."""
		try:
			# Discover all capabilities if not already done
			if not self.registry.capabilities:
				self.registry.discover_all()
			
			info = {
				'capabilities': {},
				'subcapabilities': {},
				'dependencies': [],
				'database_tables': [],
				'warnings': [],
				'total_components': 0
			}
			
			# Resolve all capabilities and dependencies
			all_capabilities = set(capabilities)
			to_process = list(capabilities)
			processed = set()
			
			while to_process:
				cap_code = to_process.pop(0)
				if cap_code in processed:
					continue
				
				processed.add(cap_code)
				capability = self.registry.get_capability(cap_code)
				
				if capability:
					info['capabilities'][cap_code] = {
						'name': capability.name,
						'version': capability.version,
						'description': capability.description,
						'subcapabilities': list(capability.subcapabilities.keys())
					}
					
					# Add sub-capabilities
					for subcap_code, subcap in capability.subcapabilities.items():
						subcap_key = f"{cap_code}.{subcap_code}"
						info['subcapabilities'][subcap_key] = {
							'name': subcap.name,
							'has_models': subcap.has_models,
							'has_api': subcap.has_api,
							'has_views': subcap.has_views,
							'database_tables': subcap.database_tables
						}
						info['database_tables'].extend(subcap.database_tables)
					
					# Add dependencies
					for dep in capability.dependencies:
						if dep.required and dep.capability not in all_capabilities:
							all_capabilities.add(dep.capability)
							to_process.append(dep.capability)
							info['dependencies'].append({
								'capability': dep.capability,
								'required_by': cap_code,
								'version_requirement': dep.version_requirement
							})
				else:
					info['warnings'].append(f"Capability '{cap_code}' not found")
			
			info['total_components'] = len(info['capabilities']) + len(info['subcapabilities'])
			
			return info
			
		except Exception as e:
			return {
				'error': f"Failed to get composition info: {str(e)}",
				'capabilities': {},
				'subcapabilities': {},
				'dependencies': [],
				'warnings': []
			}

# Global composer instance
_composer_instance: Optional[CompositionEngine] = None

def get_composer() -> CompositionEngine:
	"""Get the global composer instance."""
	global _composer_instance
	if _composer_instance is None:
		_composer_instance = CompositionEngine()
	return _composer_instance