"""
APG Connection Management Models

Core data models for connection management with Singer.io integration,
AI-powered automation, and APG platform integration.

Author: APG Platform Team
Version: 1.0.0
License: Proprietary - Datacraft © 2025
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from pydantic.types import Json
from typing_extensions import Annotated

# Use uuid7str for IDs following APG standards
def uuid7str() -> str:
	"""Generate UUID7 string for consistent ID generation."""
	return str(uuid4())  # Placeholder - in production use uuid_extensions.uuid7str

# APG Model Configuration
model_config = ConfigDict(
	extra='forbid',
	validate_by_name=True,
	validate_by_alias=True
)

class ConnectionStatus(str, Enum):
	"""Connection status enumeration"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	ERROR = "error"
	TESTING = "testing"
	CONFIGURING = "configuring"

class ConnectionType(str, Enum):
	"""Type of connection"""
	DATABASE = "database"
	API = "api"
	FILE = "file"
	STREAM = "stream"
	WEBHOOK = "webhook"
	QUEUE = "queue"

class DataFormat(str, Enum):
	"""Supported data formats"""
	JSON = "json"
	CSV = "csv"
	XML = "xml"
	PARQUET = "parquet"
	AVRO = "avro"
	PROTOBUF = "protobuf"

class SyncMode(str, Enum):
	"""Data synchronization mode"""
	FULL_REFRESH = "full_refresh"
	INCREMENTAL = "incremental"
	LOG_BASED = "log_based"
	CHANGE_DATA_CAPTURE = "change_data_capture"

class Connection(BaseModel):
	"""
	Core connection model for managing data source connections.
	Integrates with Singer.io ecosystem and APG security framework.
	"""
	model_config = model_config

	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="APG tenant identifier")
	name: str = Field(..., min_length=1, max_length=255)
	description: Optional[str] = Field(None, max_length=1000)

	# Connection Details
	connection_type: ConnectionType
	status: ConnectionStatus = ConnectionStatus.INACTIVE

	# Singer.io Integration
	singer_tap: Optional[str] = Field(None, description="Singer tap name")
	singer_target: Optional[str] = Field(None, description="Singer target name")
	tap_config: Dict[str, Any] = Field(default_factory=dict)
	target_config: Dict[str, Any] = Field(default_factory=dict)

	# Security & Authentication
	credentials_encrypted: bool = Field(default=True)
	credentials_key_id: Optional[str] = Field(None, description="APG encryption key ID")

	# Configuration
	sync_mode: SyncMode = SyncMode.INCREMENTAL
	sync_frequency: Optional[str] = Field(None, description="Cron expression")
	batch_size: int = Field(default=1000, ge=1, le=100000)

	# Monitoring & Health
	enabled: bool = Field(default=True)
	last_sync: Optional[datetime] = None
	last_success: Optional[datetime] = None
	last_error: Optional[str] = None
	error_count: int = Field(default=0, ge=0)
	records_processed: int = Field(default=0, ge=0)

	# Metadata
	tags: List[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	created_by: str = Field(default="system", description="User ID who created connection")

	def _log_connection_status(self, status: str) -> None:
		"""Log connection status changes following APG patterns."""
		print(f"Connection {self.id} status: {status}")

	async def test_connection(self) -> bool:
		"""Test connection validity with real-time feedback."""
		assert self.singer_tap, "Singer tap must be configured"
		assert self.tap_config, "Tap configuration required"

		self._log_connection_status(f"Testing connection {self.name}")

		# Simulate connection test - in production, use actual Singer tap
		await asyncio.sleep(0.1)

		if self.connection_type == ConnectionType.DATABASE:
			# Database connection test logic
			return len(self.tap_config.get('host', '')) > 0
		elif self.connection_type == ConnectionType.API:
			# API connection test logic
			return len(self.tap_config.get('api_url', '')) > 0

		return True

	async def update_status(self, status: ConnectionStatus) -> None:
		"""Update connection status with audit logging."""
		assert status in ConnectionStatus, f"Invalid status: {status}"

		old_status = self.status
		self.status = status
		self.updated_at = datetime.now(timezone.utc)

		self._log_connection_status(f"Status changed: {old_status} -> {status}")

		# Integration with APG audit logging would happen here
		# await apg_audit.log_connection_status_change(self.id, old_status, status)

class SingerTap(BaseModel):
	"""Singer.io tap configuration and metadata."""
	model_config = model_config

	id: str = Field(default_factory=uuid7str)
	name: str = Field(..., description="Singer tap name (e.g., tap-postgres)")
	display_name: str = Field(..., description="Human-readable name")
	description: str = Field(..., description="Tap description and capabilities")

	# Tap Configuration
	version: str = Field(..., description="Installed tap version")
	python_package: str = Field(..., description="PyPI package name")
	executable_path: Optional[str] = Field(None, description="Path to tap executable")

	# Schema & Capabilities
	supported_connection_types: List[ConnectionType] = Field(default_factory=list)
	supported_formats: List[DataFormat] = Field(default_factory=list)
	supports_discovery: bool = Field(default=True)
	supports_state: bool = Field(default=True)
	supports_incremental: bool = Field(default=False)

	# Configuration Schema
	config_schema: Dict[str, Any] = Field(default_factory=dict)
	required_config: List[str] = Field(default_factory=list)

	# Installation & Management
	installation_status: str = Field(default="not_installed")
	installation_date: Optional[datetime] = None
	last_used: Optional[datetime] = None
	usage_count: int = Field(default=0, ge=0)

	# APG Integration
	tenant_id: str = Field(..., description="APG tenant identifier")
	is_custom: bool = Field(default=False, description="Custom APG-developed tap")
	apg_integration: Dict[str, Any] = Field(default_factory=dict)

	# Metadata
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	def _log_tap_operation(self, operation: str) -> None:
		"""Log tap operations following APG patterns."""
		print(f"Singer tap {self.name}: {operation}")

	async def install(self) -> bool:
		"""Install Singer tap locally."""
		assert self.python_package, "Python package name required"

		self._log_tap_operation(f"Installing tap {self.name}")

		# Simulate installation - in production, use subprocess
		await asyncio.sleep(1.0)

		self.installation_status = "installed"
		self.installation_date = datetime.now(timezone.utc)

		return True

	async def discover_schema(self, connection_config: Dict[str, Any]) -> Dict[str, Any]:
		"""Discover schema for a given connection configuration."""
		assert self.supports_discovery, f"Tap {self.name} does not support discovery"
		assert connection_config, "Connection configuration required"

		self._log_tap_operation(f"Discovering schema for {self.name}")

		# Simulate schema discovery - in production, run tap in discovery mode
		await asyncio.sleep(0.5)

		return {
			"streams": [
				{
					"tap_stream_id": "users",
					"schema": {
						"type": "object",
						"properties": {
							"id": {"type": "integer"},
							"name": {"type": "string"},
							"email": {"type": "string"}
						}
					}
				}
			]
		}

class SingerTarget(BaseModel):
	"""Singer.io target configuration and metadata."""
	model_config = model_config

	id: str = Field(default_factory=uuid7str)
	name: str = Field(..., description="Singer target name (e.g., target-postgres)")
	display_name: str = Field(..., description="Human-readable name")
	description: str = Field(..., description="Target description and capabilities")

	# Target Configuration
	version: str = Field(..., description="Installed target version")
	python_package: str = Field(..., description="PyPI package name")
	executable_path: Optional[str] = Field(None, description="Path to target executable")

	# Capabilities
	supported_connection_types: List[ConnectionType] = Field(default_factory=list)
	supported_formats: List[DataFormat] = Field(default_factory=list)
	supports_hard_delete: bool = Field(default=False)
	supports_upsert: bool = Field(default=True)

	# Configuration Schema
	config_schema: Dict[str, Any] = Field(default_factory=dict)
	required_config: List[str] = Field(default_factory=list)

	# Installation & Management
	installation_status: str = Field(default="not_installed")
	installation_date: Optional[datetime] = None
	last_used: Optional[datetime] = None
	usage_count: int = Field(default=0, ge=0)

	# APG Integration
	tenant_id: str = Field(..., description="APG tenant identifier")
	is_custom: bool = Field(default=False, description="Custom APG-developed target")

	# Metadata
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DataFlow(BaseModel):
	"""Data flow definition connecting sources to destinations."""
	model_config = model_config

	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="APG tenant identifier")
	name: str = Field(..., min_length=1, max_length=255)
	description: Optional[str] = Field(None, max_length=1000)

	# Flow Configuration
	source_connection_id: Any = Field(..., description="Source connection ID")
	target_connection_id: Any = Field(..., description="Target connection ID")

	# Stream Selection
	selected_streams: List[str] = Field(default_factory=list)
	stream_config: Dict[str, Any] = Field(default_factory=dict)

	# Transformation Rules
	field_mappings: Dict[str, Any] = Field(default_factory=dict)
	transformation_config: Dict[str, Any] = Field(default_factory=dict)
	transformation_rules: List[str] = Field(default_factory=list, description="List of transformation rule IDs")

	# Scheduling
	enabled: bool = Field(default=False)
	schedule_expression: Optional[str] = Field(None, description="Cron expression")

	# State Management
	current_state: Dict[str, Any] = Field(default_factory=dict)
	last_state_update: Optional[datetime] = None

	# Execution History
	last_run: Optional[datetime] = None
	last_success: Optional[datetime] = None
	last_error: Optional[str] = None
	run_count: int = Field(default=0, ge=0)
	success_count: int = Field(default=0, ge=0)
	error_count: int = Field(default=0, ge=0)

	# Performance Metrics
	avg_runtime_seconds: float = Field(default=0.0, ge=0.0)
	avg_records_per_run: int = Field(default=0, ge=0)
	total_records_processed: int = Field(default=0, ge=0)

	# Metadata
	tags: List[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	created_by: str = Field(default="system", description="User ID who created flow")

	async def execute(self) -> Dict[str, Any]:
		"""Execute the data flow and return execution results."""
		assert self.enabled, "Flow must be enabled to execute"
		assert self.source_connection_id, "Source connection required"
		assert self.target_connection_id, "Target connection required"

		self.last_run = datetime.now(timezone.utc)
		self.run_count += 1

		try:
			# Simulate flow execution - in production, use Singer pipeline
			await asyncio.sleep(0.2)

			# Update success metrics
			self.last_success = datetime.now(timezone.utc)
			self.success_count += 1

			records_processed = 100  # Simulated
			self.total_records_processed += records_processed

			return {
				"status": "success",
				"records_processed": records_processed,
				"runtime_seconds": 0.2,
				"timestamp": self.last_success
			}

		except Exception as e:
			self.last_error = str(e)
			self.error_count += 1

			return {
				"status": "error",
				"error": str(e),
				"timestamp": datetime.now(timezone.utc)
			}

class TransformationRule(BaseModel):
	"""Data transformation rule for modifying data during flow execution."""
	model_config = model_config

	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="APG tenant identifier")
	name: str = Field(..., min_length=1, max_length=255)
	description: Optional[str] = Field(None, max_length=1000)

	# Rule Configuration
	rule_type: str = Field(..., description="Type of transformation")
	source_field: str = Field(..., description="Source field to transform")
	target_field: str = Field(..., description="Target field name")

	# Transformation Logic
	transformation_config: Dict[str, Any] = Field(default_factory=dict)
	jq_expression: Optional[str] = Field(None, description="jq transformation expression")
	python_code: Optional[str] = Field(None, description="Python transformation code")

	# Validation
	validation_rules: List[str] = Field(default_factory=list)
	required: bool = Field(default=False)

	# Metadata
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	created_by: str = Field(..., description="User ID who created rule")

	async def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
		"""Apply transformation rule to input data."""
		assert data, "Input data required"
		assert self.source_field in data, f"Source field {self.source_field} not found"

		try:
			if self.rule_type == "rename":
				# Simple field rename
				data[self.target_field] = data.pop(self.source_field)
			elif self.rule_type == "type_conversion":
				# Data type conversion
				conversion_type = self.transformation_config.get("type", "string")
				if conversion_type == "integer":
					data[self.target_field] = int(data[self.source_field])
				elif conversion_type == "float":
					data[self.target_field] = float(data[self.source_field])
				else:
					data[self.target_field] = str(data[self.source_field])
			elif self.rule_type == "custom" and self.python_code:
				# Custom Python transformation
				# In production, use safe execution environment
				exec(self.python_code)

			return data

		except Exception as e:
			# Return original data on transformation error
			return data

class ConnectionHealth(BaseModel):
	"""Connection health monitoring and diagnostics."""
	model_config = model_config

	connection_id: Any = Field(..., description="Connection ID")
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	# Health Metrics
	status: ConnectionStatus
	latency_ms: float = Field(ge=0.0)
	throughput_records_per_sec: float = Field(default=0.0, ge=0.0)
	error_rate: float = Field(default=0.0, ge=0.0, le=1.0)

	# Resource Usage
	cpu_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
	memory_usage_mb: float = Field(default=0.0, ge=0.0)
	network_io_mbps: float = Field(default=0.0, ge=0.0)

	# Connection-Specific Metrics
	connection_pool_size: int = Field(default=0, ge=0)
	active_connections: int = Field(default=0, ge=0)
	queue_depth: int = Field(default=0, ge=0)

	# Quality Metrics
	data_quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
	schema_compliance_rate: float = Field(default=1.0, ge=0.0, le=1.0)

	# Alerts
	alerts: List[str] = Field(default_factory=list)
	warnings: List[str] = Field(default_factory=list)

	def _log_health_check(self, message: str) -> None:
		"""Log health check results following APG patterns."""
		print(f"Connection health {self.connection_id}: {message}")

	def is_healthy(self) -> bool:
		"""Determine if connection is healthy based on metrics."""
		health_checks = [
			self.status == ConnectionStatus.ACTIVE,
			self.latency_ms < 1000,  # < 1 second latency
			self.error_rate < 0.01,  # < 1% error rate
			len(self.alerts) == 0    # No active alerts
		]

		is_healthy = all(health_checks)
		self._log_health_check(f"Health status: {'healthy' if is_healthy else 'unhealthy'}")

		return is_healthy

	async def run_diagnostics(self) -> Dict[str, Any]:
		"""Run comprehensive connection diagnostics."""
		self._log_health_check("Running diagnostics")

		# Simulate diagnostic checks
		await asyncio.sleep(0.1)

		diagnostics = {
			"connectivity": "ok" if self.status == ConnectionStatus.ACTIVE else "failed",
			"performance": "good" if self.latency_ms < 500 else "degraded",
			"reliability": "stable" if self.error_rate < 0.005 else "unstable",
			"resource_usage": "normal" if self.cpu_usage_percent < 80 else "high"
		}

		# Generate recommendations
		recommendations = []
		if self.latency_ms > 500:
			recommendations.append("Consider optimizing query performance")
		if self.error_rate > 0.01:
			recommendations.append("Review error logs for recurring issues")
		if self.cpu_usage_percent > 80:
			recommendations.append("Scale connection resources")

		return {
			"diagnostics": diagnostics,
			"recommendations": recommendations,
			"overall_health": self.is_healthy(),
			"timestamp": self.timestamp
		}
