#!/usr/bin/env python3
"""
APG Metadata Management - Connector Registry
Dynamic registry for metadata discovery connectors

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import inspect
from typing import Dict, Type, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from .base_connector import BaseConnector, ConnectorConfig, ConnectorType


@dataclass
class ConnectorInfo:
	"""Information about a registered connector"""
	name: str
	connector_class: Type[BaseConnector]
	connector_type: ConnectorType
	description: str
	version: str
	supported_features: List[str]
	required_params: List[str]
	optional_params: List[str]
	registered_at: datetime
	author: str = "Unknown"


class ConnectorRegistry:
	"""Registry for managing metadata discovery connectors"""
	
	def __init__(self):
		self.connectors: Dict[str, ConnectorInfo] = {}
		self._type_mapping: Dict[ConnectorType, List[str]] = {}
	
	def register(self,
		     name: str,
		     connector_class: Type[BaseConnector],
		     description: str = "",
		     version: str = "1.0.0",
		     supported_features: List[str] = None,
		     author: str = "APG System") -> bool:
		"""Register a new connector"""
		try:
			# Validate connector class
			if not issubclass(connector_class, BaseConnector):
				raise ValueError(f"Connector class must inherit from BaseConnector")
			
			# Get connector metadata
			connector_type = self._get_connector_type(connector_class)
			required_params, optional_params = self._analyze_connector_params(connector_class)
			
			# Create connector info
			connector_info = ConnectorInfo(
				name=name,
				connector_class=connector_class,
				connector_type=connector_type,
				description=description or f"{name.title()} metadata connector",
				version=version,
				supported_features=supported_features or [],
				required_params=required_params,
				optional_params=optional_params,
				registered_at=datetime.utcnow(),
				author=author
			)
			
			# Register connector
			self.connectors[name] = connector_info
			
			# Update type mapping
			if connector_type not in self._type_mapping:
				self._type_mapping[connector_type] = []
			if name not in self._type_mapping[connector_type]:
				self._type_mapping[connector_type].append(name)
			
			return True
			
		except Exception as e:
			print(f"Failed to register connector {name}: {str(e)}")
			return False
	
	def unregister(self, name: str) -> bool:
		"""Unregister a connector"""
		try:
			if name not in self.connectors:
				return False
			
			connector_info = self.connectors[name]
			
			# Remove from type mapping
			if connector_info.connector_type in self._type_mapping:
				if name in self._type_mapping[connector_info.connector_type]:
					self._type_mapping[connector_info.connector_type].remove(name)
				
				# Remove empty type mapping
				if not self._type_mapping[connector_info.connector_type]:
					del self._type_mapping[connector_info.connector_type]
			
			# Remove connector
			del self.connectors[name]
			return True
			
		except Exception as e:
			print(f"Failed to unregister connector {name}: {str(e)}")
			return False
	
	def get_connector(self, name: str) -> Optional[Type[BaseConnector]]:
		"""Get connector class by name"""
		connector_info = self.connectors.get(name)
		return connector_info.connector_class if connector_info else None
	
	def get_connector_info(self, name: str) -> Optional[ConnectorInfo]:
		"""Get detailed connector information"""
		return self.connectors.get(name)
	
	def list_connectors(self) -> List[str]:
		"""List all registered connector names"""
		return list(self.connectors.keys())
	
	def list_connectors_by_type(self, connector_type: ConnectorType) -> List[str]:
		"""List connectors by type"""
		return self._type_mapping.get(connector_type, [])
	
	def get_connector_summary(self) -> Dict[str, Any]:
		"""Get summary of all registered connectors"""
		summary = {
			"total_connectors": len(self.connectors),
			"by_type": {},
			"connectors": []
		}
		
		# Count by type
		for connector_type, connector_names in self._type_mapping.items():
			summary["by_type"][connector_type.value] = len(connector_names)
		
		# Connector details
		for name, info in self.connectors.items():
			summary["connectors"].append({
				"name": name,
				"type": info.connector_type.value,
				"description": info.description,
				"version": info.version,
				"features": info.supported_features,
				"author": info.author,
				"registered_at": info.registered_at.isoformat()
			})
		
		return summary
	
	def find_compatible_connectors(self, 
				       connection_string: str,
				       required_features: List[str] = None) -> List[str]:
		"""Find connectors compatible with connection string and features"""
		compatible = []
		required_features = required_features or []
		
		for name, info in self.connectors.items():
			# Check if connector can handle this connection string
			if self._is_connection_compatible(info, connection_string):
				# Check if connector supports required features
				if all(feature in info.supported_features for feature in required_features):
					compatible.append(name)
		
		return compatible
	
	def validate_connector_config(self, 
				      name: str,
				      config: ConnectorConfig) -> Dict[str, Any]:
		"""Validate connector configuration"""
		result = {
			"valid": False,
			"errors": [],
			"warnings": [],
			"missing_required": [],
			"unused_optional": []
		}
		
		connector_info = self.connectors.get(name)
		if not connector_info:
			result["errors"].append(f"Connector '{name}' not registered")
			return result
		
		# Check required parameters
		config_dict = config.to_dict()
		for param in connector_info.required_params:
			if param not in config_dict or config_dict[param] is None:
				result["missing_required"].append(param)
		
		if result["missing_required"]:
			result["errors"].append(f"Missing required parameters: {result['missing_required']}")
		
		# Check for unused parameters
		all_params = connector_info.required_params + connector_info.optional_params
		for param in config_dict:
			if param not in all_params and config_dict[param] is not None:
				result["unused_optional"].append(param)
		
		if result["unused_optional"]:
			result["warnings"].append(f"Unused parameters: {result['unused_optional']}")
		
		# Overall validation
		result["valid"] = len(result["errors"]) == 0
		
		return result
	
	def create_connector_instance(self, 
				      name: str,
				      config: ConnectorConfig) -> Optional[BaseConnector]:
		"""Create connector instance with configuration"""
		try:
			connector_class = self.get_connector(name)
			if not connector_class:
				raise ValueError(f"Connector '{name}' not found")
			
			# Validate configuration
			validation = self.validate_connector_config(name, config)
			if not validation["valid"]:
				raise ValueError(f"Invalid configuration: {validation['errors']}")
			
			# Create instance
			return connector_class(config)
			
		except Exception as e:
			print(f"Failed to create connector instance {name}: {str(e)}")
			return None
	
	def _get_connector_type(self, connector_class: Type[BaseConnector]) -> ConnectorType:
		"""Extract connector type from class"""
		# Try to get from class attribute
		if hasattr(connector_class, 'CONNECTOR_TYPE'):
			return connector_class.CONNECTOR_TYPE
		
		# Try to infer from class name
		class_name = connector_class.__name__.lower()
		
		if 'database' in class_name or any(db in class_name for db in ['postgres', 'mysql', 'mongo', 'oracle']):
			return ConnectorType.DATABASE
		elif 'file' in class_name or any(fmt in class_name for fmt in ['csv', 'json', 'parquet', 's3']):
			return ConnectorType.FILE
		elif 'api' in class_name or 'rest' in class_name or 'graphql' in class_name:
			return ConnectorType.API
		elif 'ml' in class_name or 'model' in class_name:
			return ConnectorType.ML_PLATFORM
		elif 'bi' in class_name or 'dashboard' in class_name:
			return ConnectorType.BI_TOOL
		elif 'stream' in class_name or 'kafka' in class_name:
			return ConnectorType.STREAMING
		else:
			return ConnectorType.CUSTOM
	
	def _analyze_connector_params(self, connector_class: Type[BaseConnector]) -> tuple[List[str], List[str]]:
		"""Analyze connector class to determine required and optional parameters"""
		required_params = []
		optional_params = []
		
		# Get constructor signature
		try:
			sig = inspect.signature(connector_class.__init__)
			
			# Analyze ConnectorConfig parameter
			for param_name, param in sig.parameters.items():
				if param_name == 'config' and param.annotation == ConnectorConfig:
					# Analyze ConnectorConfig fields
					config_fields = ConnectorConfig.__dataclass_fields__
					
					for field_name, field in config_fields.items():
						if field.default == field.default_factory:  # Required field
							if field_name in ['connection_string']:  # Core required fields
								required_params.append(field_name)
							else:
								optional_params.append(field_name)
						else:  # Optional field
							optional_params.append(field_name)
					break
			
		except Exception:
			# Fallback to basic required parameters
			required_params = ['connection_string']
			optional_params = ['username', 'password', 'host', 'port', 'database']
		
		return required_params, optional_params
	
	def _is_connection_compatible(self, connector_info: ConnectorInfo, connection_string: str) -> bool:
		"""Check if connector is compatible with connection string"""
		connection_lower = connection_string.lower()
		connector_name = connector_info.name.lower()
		
		# Direct name matching
		if connector_name in connection_lower:
			return True
		
		# Type-based matching
		type_patterns = {
			ConnectorType.DATABASE: [
				'postgresql', 'postgres', 'mysql', 'mongodb', 'mongo',
				'oracle', 'sqlserver', 'snowflake', 'redshift', 'bigquery'
			],
			ConnectorType.FILE: [
				's3://', 'gs://', 'file://', '.csv', '.json', '.parquet', '.avro'
			],
			ConnectorType.API: [
				'http://', 'https://', 'api.', 'rest', 'graphql'
			],
			ConnectorType.STREAMING: [
				'kafka://', 'stream', 'topic'
			]
		}
		
		patterns = type_patterns.get(connector_info.connector_type, [])
		return any(pattern in connection_lower for pattern in patterns)


# Global registry instance
_default_registry = ConnectorRegistry()


def get_default_registry() -> ConnectorRegistry:
	"""Get the default global connector registry"""
	return _default_registry


def register_connector(name: str,
		       connector_class: Type[BaseConnector],
		       **kwargs) -> bool:
	"""Register a connector in the default registry"""
	return _default_registry.register(name, connector_class, **kwargs)


def get_connector(name: str) -> Optional[Type[BaseConnector]]:
	"""Get a connector from the default registry"""
	return _default_registry.get_connector(name)


def list_connectors() -> List[str]:
	"""List all connectors in the default registry"""
	return _default_registry.list_connectors()