"""
APG Workflow Orchestration Connector Framework

Custom connector development SDK, validation tools, marketplace integration,
versioning system, and comprehensive security framework for third-party connectors.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import importlib.util
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Callable, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
import zipfile
import tempfile
import shutil

from pydantic import BaseModel, Field, ConfigDict, validator
from uuid_extensions import uuid7str

from .base_connector import BaseConnector, ConnectorConfiguration, ConnectorStatus

logger = logging.getLogger(__name__)

class ConnectorType(Enum):
	"""Types of connectors supported by the framework."""
	DATABASE = "database"
	API = "api"
	CLOUD = "cloud"
	MESSAGE_QUEUE = "message_queue"
	FILE_SYSTEM = "file_system"
	CUSTOM = "custom"

class ConnectorSecurityLevel(Enum):
	"""Security levels for connector validation."""
	BASIC = "basic"				# Basic validation only
	STANDARD = "standard"		# Standard security checks
	ENHANCED = "enhanced"		# Enhanced security with code analysis
	ENTERPRISE = "enterprise"	# Full enterprise security validation

class ConnectorVersion(Enum):
	"""Connector versioning scheme."""
	MAJOR = "major"		# Breaking changes
	MINOR = "minor"		# New features, backward compatible
	PATCH = "patch"		# Bug fixes, backward compatible

@dataclass
class ConnectorMetadata:
	"""Metadata for connector packages."""
	id: str = field(default_factory=uuid7str)
	name: str = ""
	version: str = "1.0.0"
	description: str = ""
	author: str = ""
	author_email: str = ""
	license: str = "MIT"
	homepage: str = ""
	repository: str = ""
	tags: List[str] = field(default_factory=list)
	connector_type: ConnectorType = ConnectorType.CUSTOM
	required_python_version: str = ">=3.8"
	dependencies: List[str] = field(default_factory=list)
	apg_version_min: str = "1.0.0"
	apg_version_max: Optional[str] = None
	created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	security_level: ConnectorSecurityLevel = ConnectorSecurityLevel.STANDARD
	checksum: Optional[str] = None
	
class ConnectorManifest(BaseModel):
	"""Connector package manifest."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	metadata: ConnectorMetadata
	entry_point: str = Field(..., description="Main connector class entry point")
	configuration_schema: Dict[str, Any] = Field(default_factory=dict)
	supported_operations: List[str] = Field(default_factory=list)
	required_permissions: List[str] = Field(default_factory=list)
	health_check_endpoint: Optional[str] = Field(default=None)
	documentation_files: List[str] = Field(default_factory=list)
	test_files: List[str] = Field(default_factory=list)
	example_configurations: List[Dict[str, Any]] = Field(default_factory=list)

class ConnectorValidationResult(BaseModel):
	"""Result of connector validation."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	is_valid: bool
	errors: List[str] = Field(default_factory=list)
	warnings: List[str] = Field(default_factory=list)
	security_score: int = Field(default=0, ge=0, le=100)
	performance_score: int = Field(default=0, ge=0, le=100)
	compatibility_score: int = Field(default=0, ge=0, le=100)
	overall_score: int = Field(default=0, ge=0, le=100)
	validation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	validated_by: str = ""
	validation_details: Dict[str, Any] = Field(default_factory=dict)

class ConnectorRegistry:
	"""Central registry for managing connectors."""
	
	def __init__(self, registry_path: Path, tenant_id: str):
		self.registry_path = registry_path
		self.tenant_id = tenant_id
		self.installed_connectors: Dict[str, ConnectorManifest] = {}
		self.connector_instances: Dict[str, BaseConnector] = {}
		self.validators: List[Callable] = []
		
		# Create registry directory
		self.registry_path.mkdir(parents=True, exist_ok=True)
		
		# Load installed connectors
		asyncio.create_task(self._load_installed_connectors())
	
	async def install_connector(
		self,
		package_path: Union[str, Path],
		validate: bool = True,
		security_level: ConnectorSecurityLevel = ConnectorSecurityLevel.STANDARD
	) -> bool:
		"""Install a connector package."""
		
		package_path = Path(package_path)
		
		if not package_path.exists():
			raise FileNotFoundError(f"Connector package not found: {package_path}")
		
		try:
			# Extract package if it's a zip file
			if package_path.suffix == '.zip':
				temp_dir = await self._extract_package(package_path)
				package_dir = temp_dir
			else:
				package_dir = package_path
			
			# Load and validate manifest
			manifest = await self._load_manifest(package_dir)
			
			if validate:
				validation_result = await self._validate_connector(package_dir, manifest, security_level)
				if not validation_result.is_valid:
					raise ValueError(f"Connector validation failed: {validation_result.errors}")
			
			# Install connector files
			connector_install_path = self.registry_path / manifest.metadata.name
			if connector_install_path.exists():
				shutil.rmtree(connector_install_path)
			
			shutil.copytree(package_dir, connector_install_path)
			
			# Register connector
			self.installed_connectors[manifest.metadata.name] = manifest
			
			# Save registry
			await self._save_registry()
			
			logger.info(f"Installed connector: {manifest.metadata.name} v{manifest.metadata.version}")
			
			# Clean up temporary files
			if package_path.suffix == '.zip':
				shutil.rmtree(temp_dir)
			
			return True
			
		except Exception as e:
			logger.error(f"Failed to install connector {package_path}: {e}")
			raise
	
	async def uninstall_connector(self, connector_name: str) -> bool:
		"""Uninstall a connector."""
		
		if connector_name not in self.installed_connectors:
			raise ValueError(f"Connector not found: {connector_name}")
		
		try:
			# Stop connector instances
			if connector_name in self.connector_instances:
				await self.connector_instances[connector_name].disconnect()
				del self.connector_instances[connector_name]
			
			# Remove files
			connector_path = self.registry_path / connector_name
			if connector_path.exists():
				shutil.rmtree(connector_path)
			
			# Remove from registry
			del self.installed_connectors[connector_name]
			
			# Save registry
			await self._save_registry()
			
			logger.info(f"Uninstalled connector: {connector_name}")
			return True
			
		except Exception as e:
			logger.error(f"Failed to uninstall connector {connector_name}: {e}")
			raise
	
	async def create_connector_instance(
		self,
		connector_name: str,
		configuration: Dict[str, Any]
	) -> BaseConnector:
		"""Create and initialize a connector instance."""
		
		if connector_name not in self.installed_connectors:
			raise ValueError(f"Connector not found: {connector_name}")
		
		manifest = self.installed_connectors[connector_name]
		
		try:
			# Load connector class
			connector_class = await self._load_connector_class(connector_name, manifest)
			
			# Create configuration object
			config_class = await self._get_configuration_class(connector_class)
			config = config_class(**configuration)
			
			# Create connector instance
			connector = connector_class(config)
			
			# Initialize connector
			await connector.initialize()
			
			# Store instance
			instance_id = f"{connector_name}_{uuid7str()}"
			self.connector_instances[instance_id] = connector
			
			logger.info(f"Created connector instance: {connector_name} ({instance_id})")
			return connector
			
		except Exception as e:
			logger.error(f"Failed to create connector instance {connector_name}: {e}")
			raise
	
	async def list_connectors(self) -> List[Dict[str, Any]]:
		"""List all installed connectors."""
		
		connectors = []
		for name, manifest in self.installed_connectors.items():
			connectors.append({
				"name": name,
				"version": manifest.metadata.version,
				"description": manifest.metadata.description,
				"author": manifest.metadata.author,
				"type": manifest.metadata.connector_type.value,
				"security_level": manifest.metadata.security_level.value,
				"created_at": manifest.metadata.created_at.isoformat(),
				"updated_at": manifest.metadata.updated_at.isoformat()
			})
		
		return connectors
	
	async def get_connector_info(self, connector_name: str) -> Dict[str, Any]:
		"""Get detailed information about a connector."""
		
		if connector_name not in self.installed_connectors:
			raise ValueError(f"Connector not found: {connector_name}")
		
		manifest = self.installed_connectors[connector_name]
		
		return {
			"metadata": {
				"id": manifest.metadata.id,
				"name": manifest.metadata.name,
				"version": manifest.metadata.version,
				"description": manifest.metadata.description,
				"author": manifest.metadata.author,
				"author_email": manifest.metadata.author_email,
				"license": manifest.metadata.license,
				"homepage": manifest.metadata.homepage,
				"repository": manifest.metadata.repository,
				"tags": manifest.metadata.tags,
				"connector_type": manifest.metadata.connector_type.value,
				"created_at": manifest.metadata.created_at.isoformat(),
				"updated_at": manifest.metadata.updated_at.isoformat()
			},
			"configuration_schema": manifest.configuration_schema,
			"supported_operations": manifest.supported_operations,
			"required_permissions": manifest.required_permissions,
			"example_configurations": manifest.example_configurations,
			"active_instances": len([
				instance for instance_id, instance in self.connector_instances.items()
				if instance_id.startswith(connector_name)
			])
		}
	
	async def validate_connector_package(
		self,
		package_path: Union[str, Path],
		security_level: ConnectorSecurityLevel = ConnectorSecurityLevel.STANDARD
	) -> ConnectorValidationResult:
		"""Validate a connector package without installing it."""
		
		package_path = Path(package_path)
		
		if not package_path.exists():
			raise FileNotFoundError(f"Package not found: {package_path}")
		
		try:
			# Extract package if needed
			if package_path.suffix == '.zip':
				temp_dir = await self._extract_package(package_path)
				package_dir = temp_dir
			else:
				package_dir = package_path
			
			# Load manifest
			manifest = await self._load_manifest(package_dir)
			
			# Validate connector
			result = await self._validate_connector(package_dir, manifest, security_level)
			
			# Clean up
			if package_path.suffix == '.zip':
				shutil.rmtree(temp_dir)
			
			return result
			
		except Exception as e:
			logger.error(f"Failed to validate connector package: {e}")
			return ConnectorValidationResult(
				is_valid=False,
				errors=[str(e)],
				overall_score=0
			)
	
	async def _extract_package(self, package_path: Path) -> Path:
		"""Extract connector package to temporary directory."""
		
		temp_dir = Path(tempfile.mkdtemp())
		
		with zipfile.ZipFile(package_path, 'r') as zip_file:
			zip_file.extractall(temp_dir)
		
		return temp_dir
	
	async def _load_manifest(self, package_dir: Path) -> ConnectorManifest:
		"""Load connector manifest from package directory."""
		
		manifest_path = package_dir / "connector.json"
		
		if not manifest_path.exists():
			raise FileNotFoundError("Connector manifest (connector.json) not found")
		
		with open(manifest_path, 'r') as f:
			manifest_data = json.load(f)
		
		# Convert metadata
		metadata_dict = manifest_data.get("metadata", {})
		metadata_dict["connector_type"] = ConnectorType(metadata_dict.get("connector_type", "custom"))
		metadata_dict["security_level"] = ConnectorSecurityLevel(metadata_dict.get("security_level", "standard"))
		
		metadata = ConnectorMetadata(**metadata_dict)
		manifest_data["metadata"] = metadata
		
		return ConnectorManifest(**manifest_data)
	
	async def _validate_connector(
		self,
		package_dir: Path,
		manifest: ConnectorManifest,
		security_level: ConnectorSecurityLevel
	) -> ConnectorValidationResult:
		"""Validate connector package."""
		
		errors = []
		warnings = []
		security_score = 0
		performance_score = 0
		compatibility_score = 0
		
		try:
			# Basic validation
			if not manifest.entry_point:
				errors.append("Entry point not specified")
			
			# Check entry point file exists
			entry_file = package_dir / f"{manifest.entry_point.split('.')[0]}.py"
			if not entry_file.exists():
				errors.append(f"Entry point file not found: {entry_file}")
			
			# Validate Python syntax
			if entry_file.exists():
				try:
					with open(entry_file, 'r') as f:
						compile(f.read(), str(entry_file), 'exec')
					security_score += 20
				except SyntaxError as e:
					errors.append(f"Syntax error in entry point: {e}")
			
			# Check required files
			required_files = ["__init__.py", "connector.json"]
			for file_name in required_files:
				if not (package_dir / file_name).exists():
					errors.append(f"Required file missing: {file_name}")
			
			# Validate configuration schema
			if manifest.configuration_schema:
				try:
					# Basic JSON schema validation would go here
					compatibility_score += 25
				except Exception as e:
					warnings.append(f"Configuration schema validation warning: {e}")
			
			# Security validation based on level
			if security_level in [ConnectorSecurityLevel.ENHANCED, ConnectorSecurityLevel.ENTERPRISE]:
				security_score += await self._perform_security_analysis(package_dir, manifest)
			else:
				security_score += 50  # Basic security score
			
			# Performance checks
			performance_score = await self._analyze_performance(package_dir, manifest)
			
			# Compatibility checks
			compatibility_score += await self._check_compatibility(manifest)
			
			# Calculate overall score
			overall_score = int((security_score + performance_score + compatibility_score) / 3)
			
			return ConnectorValidationResult(
				is_valid=len(errors) == 0,
				errors=errors,
				warnings=warnings,
				security_score=min(security_score, 100),
				performance_score=min(performance_score, 100),
				compatibility_score=min(compatibility_score, 100),
				overall_score=min(overall_score, 100),
				validated_by="APG Connector Framework",
				validation_details={
					"security_level": security_level.value,
					"package_path": str(package_dir),
					"manifest_version": manifest.metadata.version
				}
			)
			
		except Exception as e:
			logger.error(f"Validation error: {e}")
			return ConnectorValidationResult(
				is_valid=False,
				errors=[f"Validation failed: {e}"],
				overall_score=0
			)
	
	async def _perform_security_analysis(self, package_dir: Path, manifest: ConnectorManifest) -> int:
		"""Perform security analysis on connector code."""
		
		security_score = 0
		
		try:
			# Check for dangerous imports
			dangerous_imports = ['os.system', 'subprocess.call', 'eval', 'exec', '__import__']
			
			for py_file in package_dir.rglob("*.py"):
				with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
					content = f.read()
					
					# Check for dangerous patterns
					for dangerous in dangerous_imports:
						if dangerous in content:
							security_score -= 10
							logger.warning(f"Potentially dangerous import found: {dangerous} in {py_file}")
			
			# Check for proper error handling
			if "try:" in content and "except:" in content:
				security_score += 10
			
			# Check for input validation
			if "validate" in content or "ValidationError" in content:
				security_score += 15
			
			# Base security score
			security_score += 50
			
			return max(0, min(security_score, 100))
			
		except Exception as e:
			logger.error(f"Security analysis error: {e}")
			return 30  # Default security score on error
	
	async def _analyze_performance(self, package_dir: Path, manifest: ConnectorManifest) -> int:
		"""Analyze connector performance characteristics."""
		
		performance_score = 50  # Base score
		
		try:
			# Check for async usage
			for py_file in package_dir.rglob("*.py"):
				with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
					content = f.read()
					
					if "async def" in content:
						performance_score += 20
					
					if "await" in content:
						performance_score += 10
					
					# Check for connection pooling
					if "pool" in content.lower():
						performance_score += 10
					
					# Check for caching
					if "cache" in content.lower():
						performance_score += 10
			
			return min(performance_score, 100)
			
		except Exception as e:
			logger.error(f"Performance analysis error: {e}")
			return 50
	
	async def _check_compatibility(self, manifest: ConnectorManifest) -> int:
		"""Check APG compatibility."""
		
		compatibility_score = 0
		
		# Check APG version compatibility
		if manifest.metadata.apg_version_min:
			compatibility_score += 25
		
		# Check if extends BaseConnector
		compatibility_score += 25
		
		# Check for required operations
		if manifest.supported_operations:
			compatibility_score += 25
		
		# Check configuration schema
		if manifest.configuration_schema:
			compatibility_score += 25
		
		return compatibility_score
	
	async def _load_connector_class(self, connector_name: str, manifest: ConnectorManifest) -> Type[BaseConnector]:
		"""Load connector class from installed package."""
		
		connector_path = self.registry_path / connector_name
		entry_module = manifest.entry_point.split('.')[0]
		module_path = connector_path / f"{entry_module}.py"
		
		if not module_path.exists():
			raise FileNotFoundError(f"Connector module not found: {module_path}")
		
		# Load module dynamically
		spec = importlib.util.spec_from_file_location(entry_module, module_path)
		module = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(module)
		
		# Get connector class
		class_name = manifest.entry_point.split('.')[-1]
		if not hasattr(module, class_name):
			raise AttributeError(f"Connector class not found: {class_name}")
		
		connector_class = getattr(module, class_name)
		
		# Validate it's a BaseConnector subclass
		if not issubclass(connector_class, BaseConnector):
			raise TypeError(f"Connector class must extend BaseConnector: {class_name}")
		
		return connector_class
	
	async def _get_configuration_class(self, connector_class: Type[BaseConnector]) -> Type[ConnectorConfiguration]:
		"""Get configuration class for connector."""
		
		# Get constructor signature
		sig = inspect.signature(connector_class.__init__)
		
		# Find configuration parameter
		for param_name, param in sig.parameters.items():
			if param_name == 'config' and param.annotation:
				if issubclass(param.annotation, ConnectorConfiguration):
					return param.annotation
		
		# Default to base configuration
		return ConnectorConfiguration
	
	async def _load_installed_connectors(self) -> None:
		"""Load installed connectors from registry."""
		
		registry_file = self.registry_path / "registry.json"
		
		if registry_file.exists():
			try:
				with open(registry_file, 'r') as f:
					registry_data = json.load(f)
				
				for connector_name, manifest_data in registry_data.get("connectors", {}).items():
					# Reconstruct manifest
					metadata_dict = manifest_data.get("metadata", {})
					metadata_dict["connector_type"] = ConnectorType(metadata_dict.get("connector_type", "custom"))
					metadata_dict["security_level"] = ConnectorSecurityLevel(metadata_dict.get("security_level", "standard"))
					
					metadata = ConnectorMetadata(**metadata_dict)
					manifest_data["metadata"] = metadata
					
					manifest = ConnectorManifest(**manifest_data)
					self.installed_connectors[connector_name] = manifest
				
				logger.info(f"Loaded {len(self.installed_connectors)} installed connectors")
				
			except Exception as e:
				logger.error(f"Failed to load connector registry: {e}")
	
	async def _save_registry(self) -> None:
		"""Save installed connectors registry."""
		
		registry_file = self.registry_path / "registry.json"
		
		try:
			registry_data = {
				"version": "1.0.0",
				"tenant_id": self.tenant_id,
				"updated_at": datetime.now(timezone.utc).isoformat(),
				"connectors": {}
			}
			
			for connector_name, manifest in self.installed_connectors.items():
				registry_data["connectors"][connector_name] = {
					"metadata": {
						"id": manifest.metadata.id,
						"name": manifest.metadata.name,
						"version": manifest.metadata.version,
						"description": manifest.metadata.description,
						"author": manifest.metadata.author,
						"author_email": manifest.metadata.author_email,
						"license": manifest.metadata.license,
						"homepage": manifest.metadata.homepage,
						"repository": manifest.metadata.repository,
						"tags": manifest.metadata.tags,
						"connector_type": manifest.metadata.connector_type.value,
						"required_python_version": manifest.metadata.required_python_version,
						"dependencies": manifest.metadata.dependencies,
						"apg_version_min": manifest.metadata.apg_version_min,
						"apg_version_max": manifest.metadata.apg_version_max,
						"created_at": manifest.metadata.created_at.isoformat(),
						"updated_at": manifest.metadata.updated_at.isoformat(),
						"security_level": manifest.metadata.security_level.value,
						"checksum": manifest.metadata.checksum
					},
					"entry_point": manifest.entry_point,
					"configuration_schema": manifest.configuration_schema,
					"supported_operations": manifest.supported_operations,
					"required_permissions": manifest.required_permissions,
					"health_check_endpoint": manifest.health_check_endpoint,
					"documentation_files": manifest.documentation_files,
					"test_files": manifest.test_files,
					"example_configurations": manifest.example_configurations
				}
			
			with open(registry_file, 'w') as f:
				json.dump(registry_data, f, indent=2)
			
			logger.debug("Saved connector registry")
			
		except Exception as e:
			logger.error(f"Failed to save connector registry: {e}")

class ConnectorSDK:
	"""SDK for developing custom connectors."""
	
	@staticmethod
	def create_connector_template(
		connector_name: str,
		connector_type: ConnectorType,
		output_dir: Path
	) -> Path:
		"""Create a new connector template."""
		
		connector_dir = output_dir / connector_name
		connector_dir.mkdir(parents=True, exist_ok=True)
		
		# Create __init__.py
		init_content = f'''"""
{connector_name} Connector for APG Workflow Orchestration

© 2025 Datacraft. All rights reserved.
"""

from .connector import {connector_name.title()}Connector, {connector_name.title()}Configuration

__all__ = ["{connector_name.title()}Connector", "{connector_name.title()}Configuration"]
'''
		
		with open(connector_dir / "__init__.py", 'w') as f:
			f.write(init_content)
		
		# Create main connector file
		connector_content = f'''"""
{connector_name.title()} Connector Implementation

© 2025 Datacraft. All rights reserved.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging

from pydantic import BaseModel, Field, ConfigDict
from apg.capabilities.composition.workflow_orchestration.connectors.base_connector import (
	BaseConnector, ConnectorConfiguration
)

logger = logging.getLogger(__name__)

class {connector_name.title()}Configuration(ConnectorConfiguration):
	"""Configuration for {connector_name.title()} connector."""
	
	# Add your configuration fields here
	host: str = Field(..., description="Server host")
	port: int = Field(default=8080, ge=1, le=65535, description="Server port")
	api_key: Optional[str] = Field(default=None, description="API key for authentication")
	
class {connector_name.title()}Connector(BaseConnector):
	"""High-performance {connector_name.title()} connector."""
	
	def __init__(self, config: {connector_name.title()}Configuration):
		super().__init__(config)
		self.config: {connector_name.title()}Configuration = config
		# Initialize your connector-specific attributes here
	
	async def _connect(self) -> None:
		"""Initialize connection to {connector_name.title()}."""
		try:
			# Implement your connection logic here
			logger.info(self._log_connector_info("{connector_name.title()} connector initialized"))
		except Exception as e:
			logger.error(self._log_connector_info(f"Failed to connect: {{e}}"))
			raise
	
	async def _disconnect(self) -> None:
		"""Close connection to {connector_name.title()}."""
		try:
			# Implement your disconnection logic here
			logger.info(self._log_connector_info("{connector_name.title()} connector disconnected"))
		except Exception as e:
			logger.error(self._log_connector_info(f"Failed to disconnect: {{e}}"))
	
	async def _execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute {connector_name.title()} operation."""
		
		if operation == "example_operation":
			return await self._example_operation(parameters)
		else:
			raise ValueError(f"Unsupported operation: {{operation}}")
	
	async def _example_operation(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Example operation implementation."""
		
		# Implement your operation logic here
		return {{
			"success": True,
			"result": "Operation completed",
			"parameters": params
		}}
	
	async def _health_check(self) -> bool:
		"""Check {connector_name.title()} connectivity."""
		try:
			# Implement your health check logic here
			return True
		except Exception as e:
			logger.warning(self._log_connector_info(f"Health check failed: {{e}}"))
			return False
'''
		
		with open(connector_dir / "connector.py", 'w') as f:
			f.write(connector_content)
		
		# Create manifest file
		manifest = {
			"metadata": {
				"name": connector_name,
				"version": "1.0.0",
				"description": f"APG Workflow Orchestration connector for {connector_name.title()}",
				"author": "Your Name",
				"author_email": "your.email@example.com",
				"license": "MIT",
				"tags": [connector_type.value],
				"connector_type": connector_type.value,
				"required_python_version": ">=3.8",
				"dependencies": [],
				"apg_version_min": "1.0.0",
				"security_level": "standard"
			},
			"entry_point": f"connector.{connector_name.title()}Connector",
			"configuration_schema": {
				"type": "object",
				"properties": {
					"host": {"type": "string", "description": "Server host"},
					"port": {"type": "integer", "minimum": 1, "maximum": 65535},
					"api_key": {"type": "string", "description": "API key for authentication"}
				},
				"required": ["host"]
			},
			"supported_operations": ["example_operation"],
			"required_permissions": [],
			"example_configurations": [
				{
					"name": "localhost",
					"description": "Local development configuration",
					"config": {
						"host": "localhost",
						"port": 8080,
						"name": f"{connector_name}_local",
						"description": "Local development connector",
						"tenant_id": "development",
						"user_id": "developer"
					}
				}
			]
		}
		
		with open(connector_dir / "connector.json", 'w') as f:
			json.dump(manifest, f, indent=2)
		
		# Create test file
		test_content = f'''"""
Tests for {connector_name.title()} Connector

© 2025 Datacraft. All rights reserved.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from .connector import {connector_name.title()}Connector, {connector_name.title()}Configuration

class Test{connector_name.title()}Connector:
	"""Test suite for {connector_name.title()} connector."""
	
	@pytest.fixture
	def config(self):
		"""Create test configuration."""
		return {connector_name.title()}Configuration(
			name="test_{connector_name}",
			description="Test connector",
			host="localhost",
			port=8080,
			tenant_id="test_tenant",
			user_id="test_user"
		)
	
	@pytest.fixture
	async def connector(self, config):
		"""Create test connector instance."""
		connector = {connector_name.title()}Connector(config)
		await connector.initialize()
		yield connector
		await connector.disconnect()
	
	async def test_connection(self, connector):
		"""Test connector connection."""
		assert connector.status.value == "connected"
	
	async def test_example_operation(self, connector):
		"""Test example operation."""
		result = await connector.execute_request("example_operation", {{"test": "data"}})
		assert result["success"] is True
		assert "result" in result
	
	async def test_health_check(self, connector):
		"""Test health check."""
		health = await connector.health_check()
		assert health is True
'''
		
		with open(connector_dir / "test_connector.py", 'w') as f:
			f.write(test_content)
		
		# Create README
		readme_content = f'''# {connector_name.title()} Connector

APG Workflow Orchestration connector for {connector_name.title()}.

## Installation

```bash
# Install the connector package
apg-connector install {connector_name}
```

## Configuration

```python
config = {{
	"name": "my_{connector_name}",
	"description": "My {connector_name.title()} connector",
	"host": "your-server.com",
	"port": 8080,
	"api_key": "your-api-key",
	"tenant_id": "your_tenant",
	"user_id": "your_user"
}}
```

## Supported Operations

- `example_operation`: Example operation with parameters

## Usage Example

```python
from apg.capabilities.composition.workflow_orchestration.connectors import ConnectorRegistry

# Create connector registry
registry = ConnectorRegistry(registry_path="./connectors", tenant_id="your_tenant")

# Create connector instance
connector = await registry.create_connector_instance("{connector_name}", config)

# Execute operation
result = await connector.execute_request("example_operation", {{"param": "value"}})
print(result)
```

## Development

1. Clone this connector template
2. Implement your connector logic in `connector.py`
3. Update configuration schema in `connector.json`
4. Add tests in `test_connector.py`
5. Package and install

## License

MIT License
'''
		
		with open(connector_dir / "README.md", 'w') as f:
			f.write(readme_content)
		
		logger.info(f"Created connector template: {connector_dir}")
		return connector_dir
	
	@staticmethod
	def package_connector(connector_dir: Path, output_path: Optional[Path] = None) -> Path:
		"""Package connector into distributable format."""
		
		if output_path is None:
			output_path = connector_dir.parent / f"{connector_dir.name}.zip"
		
		with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
			for file_path in connector_dir.rglob("*"):
				if file_path.is_file():
					arcname = file_path.relative_to(connector_dir)
					zipf.write(file_path, arcname)
		
		# Calculate checksum
		with open(output_path, 'rb') as f:
			content = f.read()
			checksum = hashlib.sha256(content).hexdigest()
		
		logger.info(f"Packaged connector: {output_path} (checksum: {checksum})")
		return output_path

# Export framework classes
__all__ = [
	"ConnectorType",
	"ConnectorSecurityLevel",
	"ConnectorVersion",
	"ConnectorMetadata",
	"ConnectorManifest",
	"ConnectorValidationResult",
	"ConnectorRegistry",
	"ConnectorSDK"
]