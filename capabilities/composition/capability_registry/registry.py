"""
Capability Registry System

Auto-discovers all available capabilities and sub-capabilities throughout the APG system.
Builds comprehensive metadata about each capability including dependencies, interfaces,
configuration requirements, and composition compatibility.
"""

import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import asyncio
import logging
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict, validator
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, JSON, Index
from sqlalchemy.orm import relationship
from flask_appbuilder import Model

logger = logging.getLogger(__name__)

# Pydantic models for capability metadata
class DependencyInfo(BaseModel):
	"""Information about a capability dependency."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	capability: str = Field(..., description="Capability code")
	version_requirement: str = Field(default=">=1.0.0", description="Version requirement")
	required: bool = Field(default=True, description="Whether dependency is required")
	integration_points: list[str] = Field(default_factory=list, description="Integration points")
	reason: str = Field(default="", description="Reason for dependency")

class InterfaceInfo(BaseModel):
	"""Information about capability interfaces."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	name: str = Field(..., description="Interface name")
	type: str = Field(..., description="Interface type (class, function, service)")
	description: str = Field(default="", description="Interface description")
	methods: list[str] = Field(default_factory=list, description="Available methods")
	parameters: dict[str, Any] = Field(default_factory=dict, description="Interface parameters")

class SubCapabilityMetadata(BaseModel):
	"""Metadata for a sub-capability."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	code: str = Field(..., description="Sub-capability code")
	name: str = Field(..., description="Display name")
	description: str = Field(default="", description="Description")
	version: str = Field(default="1.0.0", description="Version")
	parent_capability: str = Field(..., description="Parent capability code")
	
	# Technical details
	module_path: str = Field(..., description="Python module path")
	has_models: bool = Field(default=False, description="Has database models")
	has_service: bool = Field(default=False, description="Has service layer")
	has_views: bool = Field(default=False, description="Has web views")
	has_api: bool = Field(default=False, description="Has API endpoints")
	has_blueprint: bool = Field(default=False, description="Has Flask blueprint")
	
	# Dependencies and integration
	dependencies: list[DependencyInfo] = Field(default_factory=list, description="Dependencies")
	interfaces: list[InterfaceInfo] = Field(default_factory=list, description="Provided interfaces")
	events_emitted: list[str] = Field(default_factory=list, description="Events emitted")
	events_consumed: list[str] = Field(default_factory=list, description="Events consumed")
	
	# Configuration
	configuration_schema: dict[str, Any] = Field(default_factory=dict, description="Configuration schema")
	database_tables: list[str] = Field(default_factory=list, description="Database tables")
	permissions: list[str] = Field(default_factory=list, description="Required permissions")
	
	# Metadata
	industry_focus: list[str] = Field(default_factory=list, description="Industry focus areas")
	business_processes: list[str] = Field(default_factory=list, description="Business processes supported")
	compliance_frameworks: list[str] = Field(default_factory=list, description="Compliance frameworks")
	
	discovered_at: datetime = Field(default_factory=datetime.utcnow, description="Discovery timestamp")
	file_path: str = Field(default="", description="File path where discovered")

class CapabilityMetadata(BaseModel):
	"""Metadata for a main capability."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	code: str = Field(..., description="Capability code")
	name: str = Field(..., description="Display name")
	description: str = Field(default="", description="Description")
	version: str = Field(default="1.0.0", description="Version")
	
	# Sub-capabilities
	subcapabilities: dict[str, SubCapabilityMetadata] = Field(default_factory=dict, description="Sub-capabilities")
	
	# Capability-level information
	module_path: str = Field(..., description="Python module path")
	industry_focus: list[str] = Field(default_factory=list, description="Industry focus areas")
	business_domain: str = Field(default="", description="Business domain")
	
	# Technical architecture
	composition_keywords: list[str] = Field(default_factory=list, description="Composition keywords")
	primary_interfaces: list[str] = Field(default_factory=list, description="Primary interfaces")
	event_types: list[str] = Field(default_factory=list, description="Event types")
	
	# Dependencies and integration
	dependencies: list[DependencyInfo] = Field(default_factory=list, description="Capability dependencies")
	configuration_schema: dict[str, Any] = Field(default_factory=dict, description="Configuration schema")
	
	# Metadata
	discovered_at: datetime = Field(default_factory=datetime.utcnow, description="Discovery timestamp")
	file_path: str = Field(default="", description="File path where discovered")

# Database models for persistent registry
class APGCapabilityRegistry(Model):
	"""Database model for capability registry."""
	
	__tablename__ = 'apg_capability_registry'
	
	# Identity
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True, default="system")
	
	# Capability information
	capability_code = Column(String(100), nullable=False, index=True)
	capability_name = Column(String(200), nullable=False)
	capability_version = Column(String(20), nullable=False)
	capability_type = Column(String(50), nullable=False)  # main, sub
	parent_capability = Column(String(100), nullable=True, index=True)
	
	# Technical details
	module_path = Column(String(500), nullable=False)
	file_path = Column(String(1000), nullable=True)
	
	# Capabilities flags
	has_models = Column(Boolean, default=False)
	has_service = Column(Boolean, default=False)
	has_views = Column(Boolean, default=False)
	has_api = Column(Boolean, default=False)
	has_blueprint = Column(Boolean, default=False)
	
	# Metadata
	metadata_json = Column(JSON, default=dict)
	dependencies_json = Column(JSON, default=list)
	interfaces_json = Column(JSON, default=list)
	configuration_schema = Column(JSON, default=dict)
	
	# Discovery information
	discovered_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	last_validated = Column(DateTime, nullable=True)
	is_active = Column(Boolean, default=True)
	validation_errors = Column(JSON, default=list)
	
	# Table constraints
	__table_args__ = (
		Index('ix_capability_registry_code_type', 'capability_code', 'capability_type'),
		Index('ix_capability_registry_tenant_active', 'tenant_id', 'is_active'),
	)

class CapabilityRegistry:
	"""
	Main capability registry for auto-discovery and validation.
	
	Scans the entire APG capabilities directory structure to build a comprehensive
	registry of all available capabilities and sub-capabilities.
	"""
	
	def __init__(self, capabilities_root: Optional[str] = None):
		"""Initialize the capability registry."""
		if capabilities_root is None:
			# Auto-detect capabilities root from this file's location
			current_dir = Path(__file__).parent.parent
			capabilities_root = str(current_dir)
		
		self.capabilities_root = Path(capabilities_root)
		self.capabilities: Dict[str, CapabilityMetadata] = {}
		self.subcapabilities: Dict[str, SubCapabilityMetadata] = {}
		self._discovery_cache: Dict[str, Any] = {}
		
		logger.info(f"Initialized CapabilityRegistry with root: {self.capabilities_root}")
	
	def discover_all(self, force_refresh: bool = False) -> Dict[str, CapabilityMetadata]:
		"""
		Discover all capabilities and sub-capabilities.
		
		Args:
			force_refresh: Force full re-discovery even if cached
			
		Returns:
			Dictionary mapping capability codes to metadata
		"""
		if not force_refresh and self.capabilities:
			return self.capabilities
		
		logger.info("Starting capability discovery...")
		self.capabilities.clear()
		self.subcapabilities.clear()
		
		# Scan all directories in capabilities root
		for item in self.capabilities_root.iterdir():
			if item.is_dir() and not item.name.startswith('.') and item.name != '__pycache__':
				try:
					self._discover_capability(item)
				except Exception as e:
					logger.error(f"Error discovering capability {item.name}: {e}")
		
		logger.info(f"Discovery complete. Found {len(self.capabilities)} capabilities, {len(self.subcapabilities)} sub-capabilities")
		return self.capabilities
	
	def _discover_capability(self, capability_path: Path) -> None:
		"""Discover a single capability and its sub-capabilities."""
		capability_name = capability_path.name
		
		# Skip composition directory to avoid circular imports
		if capability_name == 'composition':
			return
		
		try:
			# Try to import the capability module
			module_path = f"capabilities.{capability_name}"
			
			# Add the parent directory to sys.path temporarily
			parent_path = str(capability_path.parent.parent)
			if parent_path not in sys.path:
				sys.path.insert(0, parent_path)
			
			try:
				module = importlib.import_module(module_path)
			except ImportError as e:
				logger.warning(f"Could not import capability {capability_name}: {e}")
				return
			
			# Extract capability metadata
			capability_meta = self._extract_capability_metadata(module, capability_path)
			if capability_meta:
				self.capabilities[capability_meta.code] = capability_meta
				
				# Discover sub-capabilities
				self._discover_subcapabilities(capability_path, capability_meta.code)
			
		except Exception as e:
			logger.error(f"Error discovering capability {capability_name}: {e}")
	
	def _extract_capability_metadata(self, module, capability_path: Path) -> Optional[CapabilityMetadata]:
		"""Extract metadata from a capability module."""
		try:
			# Try different metadata extraction methods
			metadata = None
			
			# Method 1: Check for __capability_code__ style metadata (auth_rbac pattern)
			if hasattr(module, '__capability_code__'):
				metadata = CapabilityMetadata(
					code=getattr(module, '__capability_code__', capability_path.name.upper()),
					name=getattr(module, '__capability_name__', capability_path.name.replace('_', ' ').title()),
					version=getattr(module, '__version__', '1.0.0'),
					description=module.__doc__ or "",
					module_path=module.__name__,
					composition_keywords=getattr(module, '__composition_keywords__', []),
					primary_interfaces=getattr(module, '__primary_interfaces__', []),
					event_types=getattr(module, '__event_types__', []),
					dependencies=self._parse_dependencies(getattr(module, '__capability_dependencies__', [])),
					configuration_schema=getattr(module, '__configuration_schema__', {}),
					file_path=str(capability_path)
				)
			
			# Method 2: Check for CAPABILITY_META style (human_resources pattern)
			elif hasattr(module, 'CAPABILITY_META'):
				meta = module.CAPABILITY_META
				metadata = CapabilityMetadata(
					code=meta.get('code', capability_path.name.upper()),
					name=meta.get('name', capability_path.name.replace('_', ' ').title()),
					version=meta.get('version', '1.0.0'),
					description=meta.get('description', module.__doc__ or ""),
					module_path=module.__name__,
					industry_focus=meta.get('industry_focus', []) if isinstance(meta.get('industry_focus'), list) else [meta.get('industry_focus', '')],
					business_domain=meta.get('industry_focus', '') if isinstance(meta.get('industry_focus'), str) else '',
					file_path=str(capability_path)
				)
			
			# Method 3: Check for get_capability_info function
			elif hasattr(module, 'get_capability_info'):
				try:
					info = module.get_capability_info()
					metadata = CapabilityMetadata(
						code=info.get('code', capability_path.name.upper()),
						name=info.get('name', capability_path.name.replace('_', ' ').title()),
						version=info.get('version', '1.0.0'),
						description=info.get('description', module.__doc__ or ""),
						module_path=module.__name__,
						industry_focus=info.get('industry_focus', []) if isinstance(info.get('industry_focus'), list) else [info.get('industry_focus', '')],
						business_domain=info.get('industry_focus', '') if isinstance(info.get('industry_focus'), str) else '',
						file_path=str(capability_path)
					)
				except Exception as e:
					logger.warning(f"Error calling get_capability_info for {capability_path.name}: {e}")
			
			# Method 4: Fallback - create basic metadata
			if metadata is None:
				metadata = CapabilityMetadata(
					code=capability_path.name.upper(),
					name=capability_path.name.replace('_', ' ').title(),
					description=module.__doc__ or "",
					module_path=module.__name__,
					file_path=str(capability_path)
				)
			
			return metadata
			
		except Exception as e:
			logger.error(f"Error extracting capability metadata for {capability_path.name}: {e}")
			return None
	
	def _discover_subcapabilities(self, capability_path: Path, parent_code: str) -> None:
		"""Discover sub-capabilities within a capability."""
		for item in capability_path.iterdir():
			if item.is_dir() and not item.name.startswith('.') and item.name != '__pycache__':
				# Check if this looks like a sub-capability
				if self._is_subcapability(item):
					try:
						subcap_meta = self._extract_subcapability_metadata(item, parent_code)
						if subcap_meta:
							subcap_key = f"{parent_code}.{subcap_meta.code}"
							self.subcapabilities[subcap_key] = subcap_meta
							self.capabilities[parent_code].subcapabilities[subcap_meta.code] = subcap_meta
					except Exception as e:
						logger.error(f"Error discovering sub-capability {item.name}: {e}")
	
	def _is_subcapability(self, path: Path) -> bool:
		"""Check if a directory looks like a sub-capability."""
		# Look for typical sub-capability files
		subcap_indicators = ['models.py', 'service.py', 'views.py', 'api.py', 'blueprint.py']
		return any((path / indicator).exists() for indicator in subcap_indicators)
	
	def _extract_subcapability_metadata(self, subcap_path: Path, parent_code: str) -> Optional[SubCapabilityMetadata]:
		"""Extract metadata from a sub-capability."""
		try:
			subcap_name = subcap_path.name
			
			# Try to import the sub-capability module
			parent_module = parent_code.lower()
			module_path = f"capabilities.{parent_module}.{subcap_name}"
			
			try:
				module = importlib.import_module(module_path)
			except ImportError:
				# Create basic metadata without importing
				module = None
			
			# Check what files exist
			has_models = (subcap_path / 'models.py').exists()
			has_service = (subcap_path / 'service.py').exists()
			has_views = (subcap_path / 'views.py').exists()
			has_api = (subcap_path / 'api.py').exists()
			has_blueprint = (subcap_path / 'blueprint.py').exists()
			
			# Extract metadata if module is available
			description = ""
			interfaces = []
			dependencies = []
			configuration_schema = {}
			
			if module:
				description = module.__doc__ or ""
				
				# Try to extract interfaces
				for name, obj in inspect.getmembers(module):
					if inspect.isclass(obj) and not name.startswith('_'):
						interfaces.append(InterfaceInfo(
							name=name,
							type="class",
							description=obj.__doc__ or "",
							methods=[m for m, _ in inspect.getmembers(obj, inspect.ismethod)]
						))
			
			# Get database tables if models exist
			database_tables = []
			if has_models:
				database_tables = self._extract_database_tables(subcap_path / 'models.py')
			
			metadata = SubCapabilityMetadata(
				code=subcap_name,
				name=subcap_name.replace('_', ' ').title(),
				description=description,
				parent_capability=parent_code,
				module_path=module_path,
				has_models=has_models,
				has_service=has_service,
				has_views=has_views,
				has_api=has_api,
				has_blueprint=has_blueprint,
				interfaces=interfaces,
				dependencies=dependencies,
				configuration_schema=configuration_schema,
				database_tables=database_tables,
				file_path=str(subcap_path)
			)
			
			return metadata
			
		except Exception as e:
			logger.error(f"Error extracting sub-capability metadata for {subcap_path.name}: {e}")
			return None
	
	def _extract_database_tables(self, models_file: Path) -> List[str]:
		"""Extract database table names from a models.py file."""
		tables = []
		try:
			with open(models_file, 'r') as f:
				content = f.read()
				
				# Look for __tablename__ declarations
				import re
				table_matches = re.findall(r"__tablename__\s*=\s*['\"]([^'\"]+)['\"]", content)
				tables.extend(table_matches)
				
		except Exception as e:
			logger.warning(f"Error extracting tables from {models_file}: {e}")
		
		return tables
	
	def _parse_dependencies(self, deps: List[Dict[str, Any]]) -> List[DependencyInfo]:
		"""Parse dependency information from capability metadata."""
		parsed_deps = []
		
		for dep in deps:
			if isinstance(dep, dict):
				parsed_deps.append(DependencyInfo(
					capability=dep.get('capability', ''),
					version_requirement=dep.get('version', '>=1.0.0'),
					required=dep.get('required', True),
					integration_points=dep.get('integration_points', []),
					reason=dep.get('reason', '')
				))
		
		return parsed_deps
	
	def get_capability(self, code: str) -> Optional[CapabilityMetadata]:
		"""Get capability metadata by code."""
		if not self.capabilities:
			self.discover_all()
		return self.capabilities.get(code)
	
	def get_subcapability(self, capability_code: str, subcapability_code: str) -> Optional[SubCapabilityMetadata]:
		"""Get sub-capability metadata."""
		capability = self.get_capability(capability_code)
		if capability:
			return capability.subcapabilities.get(subcapability_code)
		return None
	
	def search_capabilities(self, 
						  industry: Optional[str] = None,
						  business_process: Optional[str] = None,
						  keyword: Optional[str] = None) -> List[CapabilityMetadata]:
		"""Search capabilities by various criteria."""
		if not self.capabilities:
			self.discover_all()
		
		results = []
		
		for capability in self.capabilities.values():
			matches = True
			
			if industry:
				if industry.lower() not in [i.lower() for i in capability.industry_focus]:
					matches = False
			
			if keyword:
				search_text = f"{capability.name} {capability.description} {' '.join(capability.composition_keywords)}".lower()
				if keyword.lower() not in search_text:
					matches = False
			
			if matches:
				results.append(capability)
		
		return results
	
	def get_dependencies(self, capability_code: str, recursive: bool = True) -> List[str]:
		"""Get all dependencies for a capability."""
		capability = self.get_capability(capability_code)
		if not capability:
			return []
		
		deps = set()
		
		# Direct dependencies
		for dep in capability.dependencies:
			deps.add(dep.capability)
			
			# Recursive dependencies
			if recursive:
				recursive_deps = self.get_dependencies(dep.capability, recursive=True)
				deps.update(recursive_deps)
		
		return list(deps)
	
	def get_dependents(self, capability_code: str) -> List[str]:
		"""Get capabilities that depend on the given capability."""
		if not self.capabilities:
			self.discover_all()
		
		dependents = []
		
		for cap_code, capability in self.capabilities.items():
			for dep in capability.dependencies:
				if dep.capability == capability_code:
					dependents.append(cap_code)
					break
		
		return dependents
	
	async def save_to_database(self, session) -> None:
		"""Save registry to database for persistence."""
		try:
			# Clear existing registry entries
			session.query(APGCapabilityRegistry).delete()
			
			# Save capabilities
			for capability in self.capabilities.values():
				registry_entry = APGCapabilityRegistry(
					capability_code=capability.code,
					capability_name=capability.name,
					capability_version=capability.version,
					capability_type='main',
					module_path=capability.module_path,
					file_path=capability.file_path,
					metadata_json=capability.dict(),
					dependencies_json=[dep.dict() for dep in capability.dependencies],
					configuration_schema=capability.configuration_schema
				)
				session.add(registry_entry)
				
				# Save sub-capabilities
				for subcap in capability.subcapabilities.values():
					subcap_entry = APGCapabilityRegistry(
						capability_code=subcap.code,
						capability_name=subcap.name,
						capability_version=subcap.version,
						capability_type='sub',
						parent_capability=capability.code,
						module_path=subcap.module_path,
						file_path=subcap.file_path,
						has_models=subcap.has_models,
						has_service=subcap.has_service,
						has_views=subcap.has_views,
						has_api=subcap.has_api,
						has_blueprint=subcap.has_blueprint,
						metadata_json=subcap.dict(),
						dependencies_json=[dep.dict() for dep in subcap.dependencies],
						configuration_schema=subcap.configuration_schema
					)
					session.add(subcap_entry)
			
			session.commit()
			logger.info("Registry saved to database successfully")
			
		except Exception as e:
			session.rollback()
			logger.error(f"Error saving registry to database: {e}")
			raise

# Global registry instance
_registry_instance: Optional[CapabilityRegistry] = None

def get_registry() -> CapabilityRegistry:
	"""Get the global registry instance."""
	global _registry_instance
	if _registry_instance is None:
		_registry_instance = CapabilityRegistry()
	return _registry_instance

def refresh_registry() -> CapabilityRegistry:
	"""Force refresh the global registry."""
	global _registry_instance
	_registry_instance = CapabilityRegistry()
	_registry_instance.discover_all(force_refresh=True)
	return _registry_instance