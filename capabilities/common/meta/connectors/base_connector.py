#!/usr/bin/env python3
"""
APG Metadata Management - Base Connector Framework
Abstract base connector for metadata discovery from data sources

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from uuid_extensions import uuid7str


class ConnectorType(str, Enum):
	"""Types of data source connectors"""
	DATABASE = "database"
	FILE = "file"
	API = "api"
	ML_PLATFORM = "ml_platform"
	BI_TOOL = "bi_tool"
	STREAMING = "streaming"
	CUSTOM = "custom"


class DataType(str, Enum):
	"""Standard data types for metadata"""
	STRING = "string"
	INTEGER = "integer"
	FLOAT = "float"
	BOOLEAN = "boolean"
	DATE = "date"
	DATETIME = "datetime"
	TIMESTAMP = "timestamp"
	JSON = "json"
	BINARY = "binary"
	ARRAY = "array"
	OBJECT = "object"
	UNKNOWN = "unknown"


@dataclass
class ConnectorConfig:
	"""Configuration for data source connectors"""
	connection_string: str
	username: Optional[str] = None
	password: Optional[str] = None
	host: Optional[str] = None
	port: Optional[int] = None
	database: Optional[str] = None
	schema: Optional[str] = None
	additional_params: Dict[str, Any] = field(default_factory=dict)
	
	# Discovery configuration
	include_patterns: List[str] = field(default_factory=list)
	exclude_patterns: List[str] = field(default_factory=list)
	max_sample_rows: int = 1000
	enable_profiling: bool = True
	enable_schema_inference: bool = True
	connection_timeout: int = 30
	
	# Security
	use_ssl: bool = False
	ssl_cert_path: Optional[str] = None
	ssl_key_path: Optional[str] = None
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert config to dictionary, masking sensitive data"""
		config_dict = {
			"connection_string": self._mask_connection_string(),
			"host": self.host,
			"port": self.port,
			"database": self.database,
			"schema": self.schema,
			"additional_params": self.additional_params,
			"include_patterns": self.include_patterns,
			"exclude_patterns": self.exclude_patterns,
			"max_sample_rows": self.max_sample_rows,
			"enable_profiling": self.enable_profiling,
			"enable_schema_inference": self.enable_schema_inference,
			"connection_timeout": self.connection_timeout,
			"use_ssl": self.use_ssl
		}
		return config_dict
	
	def _mask_connection_string(self) -> str:
		"""Mask sensitive information in connection string"""
		if not self.connection_string:
			return ""
		
		# Simple masking - in production use more sophisticated approach
		if '@' in self.connection_string:
			parts = self.connection_string.split('@')
			if len(parts) >= 2:
				credentials_part = parts[0]
				if ':' in credentials_part:
					user_pass = credentials_part.split(':')
					if len(user_pass) >= 2:
						masked = f"{user_pass[0]}:***@{parts[1]}"
						return masked
		
		return self.connection_string[:10] + "***" if len(self.connection_string) > 10 else "***"


@dataclass
class ColumnMetadata:
	"""Metadata for a column/field"""
	name: str
	data_type: DataType
	is_nullable: bool = True
	is_primary_key: bool = False
	is_foreign_key: bool = False
	foreign_key_table: Optional[str] = None
	foreign_key_column: Optional[str] = None
	max_length: Optional[int] = None
	precision: Optional[int] = None
	scale: Optional[int] = None
	default_value: Optional[str] = None
	description: Optional[str] = None
	
	# Data profiling information
	distinct_count: Optional[int] = None
	null_count: Optional[int] = None
	null_percentage: Optional[float] = None
	min_value: Optional[Any] = None
	max_value: Optional[Any] = None
	avg_value: Optional[Any] = None
	sample_values: List[Any] = field(default_factory=list)
	
	# Classification hints
	classification_hints: List[str] = field(default_factory=list)
	contains_pii: bool = False
	contains_phi: bool = False
	
	# Additional properties
	properties: Dict[str, Any] = field(default_factory=dict)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary representation"""
		return {
			"name": self.name,
			"data_type": self.data_type.value,
			"is_nullable": self.is_nullable,
			"is_primary_key": self.is_primary_key,
			"is_foreign_key": self.is_foreign_key,
			"foreign_key_table": self.foreign_key_table,
			"foreign_key_column": self.foreign_key_column,
			"max_length": self.max_length,
			"precision": self.precision,
			"scale": self.scale,
			"default_value": self.default_value,
			"description": self.description,
			"distinct_count": self.distinct_count,
			"null_count": self.null_count,
			"null_percentage": self.null_percentage,
			"min_value": str(self.min_value) if self.min_value is not None else None,
			"max_value": str(self.max_value) if self.max_value is not None else None,
			"avg_value": str(self.avg_value) if self.avg_value is not None else None,
			"sample_values": [str(v) for v in self.sample_values],
			"classification_hints": self.classification_hints,
			"contains_pii": self.contains_pii,
			"contains_phi": self.contains_phi,
			"properties": self.properties
		}


@dataclass
class AssetMetadata:
	"""Metadata for a discovered asset (table, file, API, etc.)"""
	name: str
	asset_type: str
	source_system: str
	schema_name: Optional[str] = None
	full_name: Optional[str] = None
	description: Optional[str] = None
	
	# Schema information
	columns: List[ColumnMetadata] = field(default_factory=list)
	column_count: int = 0
	row_count: Optional[int] = None
	size_bytes: Optional[int] = None
	
	# Timestamps
	created_at: Optional[datetime] = None
	modified_at: Optional[datetime] = None
	last_accessed: Optional[datetime] = None
	
	# Additional properties
	properties: Dict[str, Any] = field(default_factory=dict)
	tags: List[str] = field(default_factory=list)
	owner: Optional[str] = None
	location: Optional[str] = None
	
	# Quality indicators
	estimated_quality_score: Optional[float] = None
	quality_issues: List[str] = field(default_factory=list)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary representation"""
		return {
			"name": self.name,
			"asset_type": self.asset_type,
			"source_system": self.source_system,
			"schema_name": self.schema_name,
			"full_name": self.full_name,
			"description": self.description,
			"columns": [col.to_dict() for col in self.columns],
			"column_count": self.column_count,
			"row_count": self.row_count,
			"size_bytes": self.size_bytes,
			"created_at": self.created_at.isoformat() if self.created_at else None,
			"modified_at": self.modified_at.isoformat() if self.modified_at else None,
			"last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
			"properties": self.properties,
			"tags": self.tags,
			"owner": self.owner,
			"location": self.location,
			"estimated_quality_score": self.estimated_quality_score,
			"quality_issues": self.quality_issues
		}
	
	def get_schema_hash(self) -> str:
		"""Generate hash of schema structure for change detection"""
		schema_data = {
			"name": self.name,
			"columns": [(col.name, col.data_type.value, col.is_nullable) for col in self.columns]
		}
		schema_json = json.dumps(schema_data, sort_keys=True)
		return hashlib.sha256(schema_json.encode()).hexdigest()


@dataclass
class DiscoveryResult:
	"""Result of metadata discovery operation"""
	connector_type: ConnectorType
	source_system: str
	discovery_id: str = field(default_factory=uuid7str)
	start_time: datetime = field(default_factory=datetime.utcnow)
	end_time: Optional[datetime] = None
	
	# Results
	assets: List[AssetMetadata] = field(default_factory=list)
	total_assets: int = 0
	successful_assets: int = 0
	failed_assets: int = 0
	
	# Errors and warnings
	errors: List[str] = field(default_factory=list)
	warnings: List[str] = field(default_factory=list)
	
	# Performance metrics
	discovery_duration_seconds: Optional[float] = None
	assets_per_second: Optional[float] = None
	
	def complete_discovery(self):
		"""Mark discovery as completed and calculate metrics"""
		self.end_time = datetime.utcnow()
		self.total_assets = len(self.assets)
		self.discovery_duration_seconds = (self.end_time - self.start_time).total_seconds()
		
		if self.discovery_duration_seconds > 0:
			self.assets_per_second = self.total_assets / self.discovery_duration_seconds
	
	def add_error(self, error: str):
		"""Add error to the result"""
		self.errors.append(f"[{datetime.utcnow().isoformat()}] {error}")
		self.failed_assets += 1
	
	def add_warning(self, warning: str):
		"""Add warning to the result"""
		self.warnings.append(f"[{datetime.utcnow().isoformat()}] {warning}")
	
	def add_asset(self, asset: AssetMetadata):
		"""Add successfully discovered asset"""
		self.assets.append(asset)
		self.successful_assets += 1
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary representation"""
		return {
			"discovery_id": self.discovery_id,
			"connector_type": self.connector_type.value,
			"source_system": self.source_system,
			"start_time": self.start_time.isoformat(),
			"end_time": self.end_time.isoformat() if self.end_time else None,
			"assets": [asset.to_dict() for asset in self.assets],
			"total_assets": self.total_assets,
			"successful_assets": self.successful_assets,
			"failed_assets": self.failed_assets,
			"errors": self.errors,
			"warnings": self.warnings,
			"discovery_duration_seconds": self.discovery_duration_seconds,
			"assets_per_second": self.assets_per_second
		}


class BaseConnector(ABC):
	"""Abstract base class for all metadata connectors"""
	
	def __init__(self, config: ConnectorConfig):
		self.config = config
		self.connector_type = ConnectorType.CUSTOM
		self.source_system = "unknown"
		self.is_connected = False
		self.connection = None
	
	@abstractmethod
	async def connect(self) -> bool:
		"""Establish connection to data source"""
		pass
	
	@abstractmethod
	async def disconnect(self):
		"""Close connection to data source"""
		pass
	
	@abstractmethod
	async def test_connection(self) -> Dict[str, Any]:
		"""Test connection to data source"""
		pass
	
	@abstractmethod
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover all metadata assets from the data source"""
		pass
	
	@abstractmethod
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get detailed schema information for a specific asset"""
		pass
	
	@abstractmethod
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Sample data from an asset for profiling"""
		pass
	
	async def profile_column(self, 
						    asset_name: str, 
						    column_name: str,
						    sample_data: List[Any]) -> ColumnMetadata:
		"""Profile a column to extract metadata and statistics"""
		if not sample_data:
			return ColumnMetadata(
				name=column_name,
				data_type=DataType.UNKNOWN
			)
		
		# Infer data type from sample
		data_type = self._infer_data_type(sample_data)
		
		# Calculate statistics
		non_null_values = [v for v in sample_data if v is not None]
		null_count = len(sample_data) - len(non_null_values)
		null_percentage = (null_count / len(sample_data)) * 100 if sample_data else 0
		
		distinct_values = set(str(v) for v in non_null_values)
		distinct_count = len(distinct_values)
		
		# Min/max values for numeric and date types
		min_value = None
		max_value = None
		avg_value = None
		
		if data_type in [DataType.INTEGER, DataType.FLOAT]:
			try:
				numeric_values = [float(v) for v in non_null_values if v is not None]
				if numeric_values:
					min_value = min(numeric_values)
					max_value = max(numeric_values)
					avg_value = sum(numeric_values) / len(numeric_values)
			except (ValueError, TypeError):
				pass
		
		# Sample values for inspection (up to 10)
		sample_values = list(distinct_values)[:10]
		
		# Classification hints based on column name and content
		classification_hints = self._get_classification_hints(column_name, sample_values)
		contains_pii = self._detect_pii(column_name, sample_values)
		contains_phi = self._detect_phi(column_name, sample_values)
		
		return ColumnMetadata(
			name=column_name,
			data_type=data_type,
			distinct_count=distinct_count,
			null_count=null_count,
			null_percentage=round(null_percentage, 2),
			min_value=min_value,
			max_value=max_value,
			avg_value=avg_value,
			sample_values=sample_values,
			classification_hints=classification_hints,
			contains_pii=contains_pii,
			contains_phi=contains_phi
		)
	
	def _infer_data_type(self, sample_data: List[Any]) -> DataType:
		"""Infer data type from sample data"""
		if not sample_data:
			return DataType.UNKNOWN
		
		# Remove null values for type inference
		non_null_values = [v for v in sample_data if v is not None]
		if not non_null_values:
			return DataType.UNKNOWN
		
		# Check for boolean
		if all(isinstance(v, bool) for v in non_null_values):
			return DataType.BOOLEAN
		
		# Check for integer
		if all(isinstance(v, int) and not isinstance(v, bool) for v in non_null_values):
			return DataType.INTEGER
		
		# Check for float
		if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null_values):
			return DataType.FLOAT
		
		# Check for datetime
		if all(isinstance(v, datetime) for v in non_null_values):
			return DataType.DATETIME
		
		# Check for date strings
		date_patterns = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']
		for pattern in date_patterns:
			try:
				if all(datetime.strptime(str(v), pattern) for v in non_null_values[:5]):
					return DataType.DATE
			except (ValueError, TypeError):
				continue
		
		# Check for JSON strings
		try:
			json_count = 0
			for v in non_null_values[:10]:
				try:
					json.loads(str(v))
					json_count += 1
				except (ValueError, TypeError):
					pass
			if json_count > len(non_null_values) * 0.5:  # More than 50% are valid JSON
				return DataType.JSON
		except Exception:
			pass
		
		# Default to string
		return DataType.STRING
	
	def _get_classification_hints(self, column_name: str, sample_values: List[str]) -> List[str]:
		"""Get classification hints based on column name and content"""
		hints = []
		column_lower = column_name.lower()
		
		# Check column name patterns
		if any(pattern in column_lower for pattern in ['email', 'e_mail', 'mail']):
			hints.append('email')
		if any(pattern in column_lower for pattern in ['phone', 'telephone', 'mobile', 'cell']):
			hints.append('phone')
		if any(pattern in column_lower for pattern in ['ssn', 'social_security', 'tax_id']):
			hints.append('ssn')
		if any(pattern in column_lower for pattern in ['credit_card', 'card_number', 'cc_num']):
			hints.append('credit_card')
		if any(pattern in column_lower for pattern in ['password', 'pwd', 'secret']):
			hints.append('password')
		if any(pattern in column_lower for pattern in ['address', 'addr', 'street', 'zip', 'postal']):
			hints.append('address')
		
		# Check sample values for patterns
		if sample_values:
			# Email pattern
			if any('@' in str(v) and '.' in str(v) for v in sample_values[:5]):
				hints.append('email')
			
			# Phone number pattern
			if any(len(str(v).replace('-', '').replace('(', '').replace(')', '').replace(' ', '')) >= 10 
				  and str(v).replace('-', '').replace('(', '').replace(')', '').replace(' ', '').isdigit() 
				  for v in sample_values[:5]):
				hints.append('phone')
		
		return list(set(hints))
	
	def _detect_pii(self, column_name: str, sample_values: List[str]) -> bool:
		"""Detect if column contains PII"""
		pii_indicators = [
			'email', 'phone', 'ssn', 'social_security', 'tax_id', 'credit_card',
			'name', 'first_name', 'last_name', 'address', 'street', 'zip',
			'postal_code', 'birth_date', 'dob', 'driver_license', 'passport'
		]
		
		column_lower = column_name.lower()
		return any(indicator in column_lower for indicator in pii_indicators)
	
	def _detect_phi(self, column_name: str, sample_values: List[str]) -> bool:
		"""Detect if column contains PHI (Protected Health Information)"""
		phi_indicators = [
			'patient', 'medical', 'diagnosis', 'treatment', 'medication',
			'doctor', 'physician', 'hospital', 'clinic', 'health',
			'insurance', 'claim', 'procedure', 'icd', 'cpt'
		]
		
		column_lower = column_name.lower()
		return any(indicator in column_lower for indicator in phi_indicators)
	
	def _estimate_quality_score(self, asset: AssetMetadata) -> float:
		"""Estimate basic quality score for discovered asset"""
		score = 100.0
		
		# Deduct points for missing metadata
		if not asset.description:
			score -= 10
		
		if not asset.columns:
			score -= 20
		
		# Check column quality
		for column in asset.columns:
			if column.null_percentage and column.null_percentage > 50:
				score -= 5  # High null percentage
			
			if column.data_type == DataType.UNKNOWN:
				score -= 3  # Unknown data type
		
		# Check for naming conventions
		if '_' not in asset.name and not asset.name.islower():
			score -= 5  # Poor naming convention
		
		return max(0.0, min(100.0, score))
	
	async def _log_info(self, message: str):
		"""Log info message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] {self.__class__.__name__} INFO: {message}")
	
	async def _log_error(self, message: str):
		"""Log error message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] {self.__class__.__name__} ERROR: {message}")


# Utility functions for pattern matching

def matches_pattern(name: str, patterns: List[str]) -> bool:
	"""Check if name matches any of the given patterns"""
	import fnmatch
	
	for pattern in patterns:
		if fnmatch.fnmatch(name.lower(), pattern.lower()):
			return True
	return False


def should_include_asset(name: str, include_patterns: List[str], exclude_patterns: List[str]) -> bool:
	"""Determine if asset should be included based on include/exclude patterns"""
	# If no include patterns, include by default
	include = not include_patterns or matches_pattern(name, include_patterns)
	
	# Exclude if matches exclude patterns
	exclude = matches_pattern(name, exclude_patterns)
	
	return include and not exclude