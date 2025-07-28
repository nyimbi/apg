"""
APG Customer Relationship Management - API Versioning and Deprecation Management

This module provides comprehensive API versioning and deprecation capabilities including
version routing, backward compatibility, migration assistance, deprecation warnings,
and lifecycle management for seamless API evolution.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from enum import Enum
from dataclasses import dataclass, field
from packaging import version as pkg_version

from pydantic import BaseModel, Field, validator
from uuid_extensions import uuid7str

from .views import CRMResponse, CRMError


logger = logging.getLogger(__name__)


class APIVersionStatus(str, Enum):
	"""API version lifecycle status"""
	DEVELOPMENT = "development"
	BETA = "beta"
	STABLE = "stable"
	DEPRECATED = "deprecated"
	SUNSET = "sunset"
	RETIRED = "retired"


class DeprecationSeverity(str, Enum):
	"""Deprecation warning severity levels"""
	INFO = "info"
	WARNING = "warning"
	CRITICAL = "critical"
	BREAKING = "breaking"


class VersioningStrategy(str, Enum):
	"""API versioning strategies"""
	URL_PATH = "url_path"  # /v1/contacts
	QUERY_PARAMETER = "query_parameter"  # ?version=1.0
	HEADER = "header"  # X-API-Version: 1.0
	CONTENT_TYPE = "content_type"  # application/vnd.api+json;version=1.0
	SUBDOMAIN = "subdomain"  # v1.api.example.com
	CUSTOM = "custom"


class MigrationComplexity(str, Enum):
	"""Migration complexity levels"""
	TRIVIAL = "trivial"  # No changes required
	SIMPLE = "simple"  # Minor parameter changes
	MODERATE = "moderate"  # Some restructuring needed
	COMPLEX = "complex"  # Significant changes required  
	BREAKING = "breaking"  # Complete rewrite needed


class APIVersion(BaseModel):
	"""API version definition"""
	id: str = Field(default_factory=uuid7str)
	version_number: str  # e.g., "1.0", "2.1", "3.0-beta"
	version_name: Optional[str] = None  # e.g., "Genesis", "Evolution"
	semantic_version: str  # e.g., "1.0.0", "2.1.5"
	
	# Status and lifecycle
	status: APIVersionStatus = APIVersionStatus.DEVELOPMENT
	release_date: Optional[datetime] = None
	deprecation_date: Optional[datetime] = None
	sunset_date: Optional[datetime] = None
	retirement_date: Optional[datetime] = None
	
	# Version metadata
	description: Optional[str] = None
	changelog: List[str] = Field(default_factory=list)
	breaking_changes: List[str] = Field(default_factory=list)
	new_features: List[str] = Field(default_factory=list)
	bug_fixes: List[str] = Field(default_factory=list)
	
	# Compatibility
	backward_compatible: bool = True
	supported_versions: List[str] = Field(default_factory=list)  # Versions this can serve
	migration_paths: Dict[str, str] = Field(default_factory=dict)  # from_version -> migration_guide
	
	# Technical details
	schema_version: str = "1.0"
	api_endpoints: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
	supported_formats: List[str] = Field(default_factory=lambda: ["json"])
	authentication_methods: List[str] = Field(default_factory=list)
	
	# Usage and analytics
	usage_stats: Dict[str, Any] = Field(default_factory=dict)
	client_adoption: Dict[str, int] = Field(default_factory=dict)  # client_id -> request_count
	
	# Configuration
	is_default: bool = False
	is_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


class DeprecationNotice(BaseModel):
	"""API deprecation notice"""
	id: str = Field(default_factory=uuid7str)
	version_id: str
	endpoint_pattern: Optional[str] = None  # Specific endpoint or * for all
	
	# Deprecation details
	severity: DeprecationSeverity = DeprecationSeverity.WARNING
	title: str
	message: str
	deprecation_date: datetime
	sunset_date: Optional[datetime] = None
	
	# Migration guidance
	replacement_endpoint: Optional[str] = None
	migration_guide_url: Optional[str] = None
	migration_complexity: MigrationComplexity = MigrationComplexity.SIMPLE
	estimated_migration_effort: Optional[str] = None  # e.g., "2-4 hours"
	
	# Communication
	announcement_channels: List[str] = Field(default_factory=list)
	notification_sent: bool = False
	acknowledgment_required: bool = False
	
	# Tracking
	affected_clients: List[str] = Field(default_factory=list)
	client_acknowledgments: Dict[str, datetime] = Field(default_factory=dict)
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


class VersionMigration(BaseModel):
	"""Version migration definition"""
	id: str = Field(default_factory=uuid7str)
	from_version: str
	to_version: str
	
	# Migration details
	migration_name: str
	description: str
	complexity: MigrationComplexity = MigrationComplexity.SIMPLE
	estimated_effort: Optional[str] = None
	
	# Migration steps
	migration_steps: List[Dict[str, Any]] = Field(default_factory=list)
	validation_rules: List[Dict[str, Any]] = Field(default_factory=list)
	rollback_steps: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Field mappings
	field_mappings: Dict[str, str] = Field(default_factory=dict)  # old_field -> new_field
	deprecated_fields: List[str] = Field(default_factory=list)
	new_fields: List[str] = Field(default_factory=list)
	
	# Transformation rules
	data_transformations: List[Dict[str, Any]] = Field(default_factory=list)
	custom_transformers: Dict[str, str] = Field(default_factory=dict)
	
	# Documentation
	migration_guide_url: Optional[str] = None
	code_examples: Dict[str, str] = Field(default_factory=dict)  # language -> example
	
	# Automation
	automated_migration: bool = False
	migration_script_url: Optional[str] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


class ClientVersionUsage(BaseModel):
	"""Client API version usage tracking"""
	id: str = Field(default_factory=uuid7str)
	client_id: str
	client_name: Optional[str] = None
	
	# Version usage
	version_usage: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # version -> stats
	primary_version: Optional[str] = None
	fallback_versions: List[str] = Field(default_factory=list)
	
	# Client details
	client_type: str = "external"  # external, internal, partner
	contact_info: Dict[str, str] = Field(default_factory=dict)
	sla_tier: Optional[str] = None
	
	# Migration tracking
	migration_status: Dict[str, str] = Field(default_factory=dict)  # version -> status
	migration_deadlines: Dict[str, datetime] = Field(default_factory=dict)
	support_level: str = "standard"  # standard, premium, enterprise
	
	# Analytics
	last_request_at: Optional[datetime] = None
	total_requests: int = 0
	error_rate: float = 0.0
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class APIVersioningManager:
	"""Comprehensive API versioning and deprecation management system"""
	
	def __init__(self, db_pool, config: Optional[Dict[str, Any]] = None):
		self.db_pool = db_pool
		self.config = config or {}
		
		# Version management
		self.active_versions = {}
		self.default_version = None
		self.version_routing = {}
		self.deprecation_notices = {}
		
		# Migration management
		self.migration_paths = {}
		self.migration_handlers = {}
		self.transformation_functions = {}
		
		# Client tracking
		self.client_usage = {}
		self.version_analytics = {}
		
		# Configuration
		self.versioning_strategy = VersioningStrategy(
			self.config.get('versioning_strategy', VersioningStrategy.URL_PATH.value)
		)
		self.default_deprecation_period = self.config.get('default_deprecation_period_days', 180)
		self.sunset_grace_period = self.config.get('sunset_grace_period_days', 90)
		
		# Notification system
		self.notification_handlers = {}
		self.deprecation_scheduler = None
		
		# Request processing
		self.version_extractors = {
			VersioningStrategy.URL_PATH: self._extract_version_from_path,
			VersioningStrategy.HEADER: self._extract_version_from_header,
			VersioningStrategy.QUERY_PARAMETER: self._extract_version_from_query,
			VersioningStrategy.CONTENT_TYPE: self._extract_version_from_content_type
		}

	async def initialize(self) -> None:
		"""Initialize the API versioning manager"""
		try:
			logger.info("🔄 Initializing API versioning manager...")
			
			# Load active versions
			await self._load_active_versions()
			
			# Load deprecation notices
			await self._load_deprecation_notices()
			
			# Load migration paths
			await self._load_migration_paths()
			
			# Load client usage data
			await self._load_client_usage()
			
			# Initialize transformation functions
			await self._initialize_transformations()
			
			# Start deprecation scheduler
			await self._start_deprecation_scheduler()
			
			logger.info("✅ API versioning manager initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize API versioning manager: {str(e)}")
			raise CRMError(f"API versioning manager initialization failed: {str(e)}")

	async def create_api_version(
		self,
		version_number: str,
		semantic_version: str,
		created_by: str,
		description: Optional[str] = None,
		version_name: Optional[str] = None,
		status: APIVersionStatus = APIVersionStatus.DEVELOPMENT,
		backward_compatible: bool = True,
		changelog: Optional[List[str]] = None,
		breaking_changes: Optional[List[str]] = None,
		**kwargs
	) -> APIVersion:
		"""Create a new API version"""
		try:
			# Validate semantic version
			try:
				pkg_version.Version(semantic_version)
			except Exception:
				raise CRMError(f"Invalid semantic version: {semantic_version}")
			
			# Check if version already exists
			existing = await self._get_version_by_number(version_number)
			if existing:
				raise CRMError(f"Version {version_number} already exists")
			
			api_version = APIVersion(
				version_number=version_number,
				semantic_version=semantic_version,
				version_name=version_name,
				description=description,
				status=status,
				backward_compatible=backward_compatible,
				changelog=changelog or [],
				breaking_changes=breaking_changes or [],
				created_by=created_by,
				**kwargs
			)
			
			# Set release date if stable
			if status == APIVersionStatus.STABLE:
				api_version.release_date = datetime.utcnow()
			
			# Save to database
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_api_versions (
						id, version_number, version_name, semantic_version,
						status, release_date, deprecation_date, sunset_date,
						retirement_date, description, changelog, breaking_changes,
						new_features, bug_fixes, backward_compatible,
						supported_versions, migration_paths, schema_version,
						api_endpoints, supported_formats, authentication_methods,
						usage_stats, client_adoption, is_default, is_active,
						created_at, created_by
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27)
				""",
				api_version.id, api_version.version_number, api_version.version_name,
				api_version.semantic_version, api_version.status.value,
				api_version.release_date, api_version.deprecation_date,
				api_version.sunset_date, api_version.retirement_date,
				api_version.description, json.dumps(api_version.changelog),
				json.dumps(api_version.breaking_changes), json.dumps(api_version.new_features),
				json.dumps(api_version.bug_fixes), api_version.backward_compatible,
				json.dumps(api_version.supported_versions), json.dumps(api_version.migration_paths),
				api_version.schema_version, json.dumps(api_version.api_endpoints),
				json.dumps(api_version.supported_formats), json.dumps(api_version.authentication_methods),
				json.dumps(api_version.usage_stats), json.dumps(api_version.client_adoption),
				api_version.is_default, api_version.is_active, api_version.created_at,
				api_version.created_by)
			
			# Cache the version
			self.active_versions[version_number] = api_version
			
			# Set as default if first version or explicitly requested
			if api_version.is_default or not self.default_version:
				self.default_version = version_number
			
			logger.info(f"Created API version: {version_number} ({semantic_version})")
			return api_version
			
		except Exception as e:
			logger.error(f"Failed to create API version: {str(e)}")
			raise CRMError(f"Failed to create API version: {str(e)}")

	async def deprecate_version(
		self,
		version_number: str,
		deprecation_date: Optional[datetime] = None,
		sunset_date: Optional[datetime] = None,
		deprecation_reason: str = "Version superseded by newer release",
		replacement_version: Optional[str] = None,
		created_by: str = "system"
	) -> DeprecationNotice:
		"""Deprecate an API version"""
		try:
			version = self.active_versions.get(version_number)
			if not version:
				raise CRMError(f"Version {version_number} not found")
			
			if version.status == APIVersionStatus.RETIRED:
				raise CRMError(f"Version {version_number} is already retired")
			
			# Set deprecation and sunset dates
			if not deprecation_date:
				deprecation_date = datetime.utcnow()
			
			if not sunset_date:
				sunset_date = deprecation_date + timedelta(days=self.default_deprecation_period)
			
			# Update version status
			version.status = APIVersionStatus.DEPRECATED
			version.deprecation_date = deprecation_date
			version.sunset_date = sunset_date
			
			# Create deprecation notice
			notice = DeprecationNotice(
				version_id=version.id,
				title=f"API Version {version_number} Deprecated",
				message=deprecation_reason,
				deprecation_date=deprecation_date,
				sunset_date=sunset_date,
				replacement_endpoint=replacement_version,
				created_by=created_by
			)
			
			# Save to database
			async with self.db_pool.acquire() as conn:
				# Update version
				await conn.execute("""
					UPDATE crm_api_versions 
					SET status = $1, deprecation_date = $2, sunset_date = $3
					WHERE version_number = $4
				""", version.status.value, version.deprecation_date, 
				version.sunset_date, version_number)
				
				# Insert deprecation notice
				await conn.execute("""
					INSERT INTO crm_deprecation_notices (
						id, version_id, endpoint_pattern, severity, title,
						message, deprecation_date, sunset_date, replacement_endpoint,
						migration_guide_url, migration_complexity, estimated_migration_effort,
						announcement_channels, notification_sent, acknowledgment_required,
						affected_clients, client_acknowledgments, created_at, created_by
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
				""",
				notice.id, notice.version_id, notice.endpoint_pattern,
				notice.severity.value, notice.title, notice.message,
				notice.deprecation_date, notice.sunset_date, notice.replacement_endpoint,
				notice.migration_guide_url, notice.migration_complexity.value,
				notice.estimated_migration_effort, json.dumps(notice.announcement_channels),
				notice.notification_sent, notice.acknowledgment_required,
				json.dumps(notice.affected_clients), json.dumps(notice.client_acknowledgments),
				notice.created_at, notice.created_by)
			
			# Cache deprecation notice
			self.deprecation_notices[version_number] = notice
			
			# Send notifications to affected clients
			await self._notify_affected_clients(notice)
			
			logger.info(f"Deprecated API version: {version_number}")
			return notice
			
		except Exception as e:
			logger.error(f"Failed to deprecate version: {str(e)}")
			raise CRMError(f"Failed to deprecate version: {str(e)}")

	async def create_migration_path(
		self,
		from_version: str,
		to_version: str,
		migration_name: str,
		description: str,
		created_by: str,
		complexity: MigrationComplexity = MigrationComplexity.SIMPLE,
		field_mappings: Optional[Dict[str, str]] = None,
		migration_steps: Optional[List[Dict[str, Any]]] = None,
		**kwargs
	) -> VersionMigration:
		"""Create a migration path between versions"""
		try:
			migration = VersionMigration(
				from_version=from_version,
				to_version=to_version,
				migration_name=migration_name,
				description=description,
				complexity=complexity,
				field_mappings=field_mappings or {},
				migration_steps=migration_steps or [],
				created_by=created_by,
				**kwargs
			)
			
			# Save to database
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_version_migrations (
						id, from_version, to_version, migration_name, description,
						complexity, estimated_effort, migration_steps, validation_rules,
						rollback_steps, field_mappings, deprecated_fields, new_fields,
						data_transformations, custom_transformers, migration_guide_url,
						code_examples, automated_migration, migration_script_url,
						created_at, created_by
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
				""",
				migration.id, migration.from_version, migration.to_version,
				migration.migration_name, migration.description, migration.complexity.value,
				migration.estimated_effort, json.dumps(migration.migration_steps),
				json.dumps(migration.validation_rules), json.dumps(migration.rollback_steps),
				json.dumps(migration.field_mappings), json.dumps(migration.deprecated_fields),
				json.dumps(migration.new_fields), json.dumps(migration.data_transformations),
				json.dumps(migration.custom_transformers), migration.migration_guide_url,
				json.dumps(migration.code_examples), migration.automated_migration,
				migration.migration_script_url, migration.created_at, migration.created_by)
			
			# Cache migration path
			migration_key = f"{from_version}->{to_version}"
			self.migration_paths[migration_key] = migration
			
			logger.info(f"Created migration path: {from_version} -> {to_version}")
			return migration
			
		except Exception as e:
			logger.error(f"Failed to create migration path: {str(e)}")
			raise CRMError(f"Failed to create migration path: {str(e)}")

	async def get_api_versions(
		self,
		status: Optional[APIVersionStatus] = None,
		include_retired: bool = False
	) -> List[Dict[str, Any]]:
		"""Get API versions"""
		try:
			async with self.db_pool.acquire() as conn:
				query = "SELECT * FROM crm_api_versions WHERE 1=1"
				params = []
				
				if status:
					query += " AND status = $1"
					params.append(status.value)
				
				if not include_retired:
					query += f" AND status != $${len(params) + 1}"
					params.append(APIVersionStatus.RETIRED.value)
				
				query += " ORDER BY created_at DESC"
				
				rows = await conn.fetch(query, *params)
				return [dict(row) for row in rows]
				
		except Exception as e:
			logger.error(f"Failed to get API versions: {str(e)}")
			raise CRMError(f"Failed to get API versions: {str(e)}")

	async def get_version_usage_analytics(
		self,
		version_number: Optional[str] = None,
		days: int = 30
	) -> Dict[str, Any]:
		"""Get version usage analytics"""
		try:
			start_date = datetime.utcnow() - timedelta(days=days)
			
			async with self.db_pool.acquire() as conn:
				if version_number:
					# Specific version analytics
					usage_data = await conn.fetchrow("""
						SELECT 
							version_number,
							SUM(total_requests) as total_requests,
							AVG(avg_response_time_ms) as avg_response_time,
							COUNT(DISTINCT client_id) as unique_clients,
							MAX(last_request_at) as last_used
						FROM crm_client_version_usage 
						WHERE version_number = $1 AND updated_at >= $2
						GROUP BY version_number
					""", version_number, start_date)
				else:
					# All versions analytics
					usage_data = await conn.fetch("""
						SELECT 
							version_number,
							SUM(total_requests) as total_requests,
							AVG(avg_response_time_ms) as avg_response_time,
							COUNT(DISTINCT client_id) as unique_clients,
							MAX(last_request_at) as last_used
						FROM crm_client_version_usage 
						WHERE updated_at >= $1
						GROUP BY version_number
						ORDER BY total_requests DESC
					""", start_date)
				
				# Get deprecation status
				deprecation_data = await conn.fetch("""
					SELECT v.version_number, v.status, v.deprecation_date, v.sunset_date,
						   COUNT(dn.id) as deprecation_notices
					FROM crm_api_versions v
					LEFT JOIN crm_deprecation_notices dn ON v.id = dn.version_id
					GROUP BY v.version_number, v.status, v.deprecation_date, v.sunset_date
				""")
				
				return {
					"usage_data": [dict(row) for row in (usage_data if isinstance(usage_data, list) else [usage_data] if usage_data else [])],
					"deprecation_status": [dict(row) for row in deprecation_data],
					"period_days": days,
					"generated_at": datetime.utcnow().isoformat()
				}
				
		except Exception as e:
			logger.error(f"Failed to get version usage analytics: {str(e)}")
			raise CRMError(f"Failed to get version usage analytics: {str(e)}")

	async def extract_version_from_request(self, request_info: Dict[str, Any]) -> str:
		"""Extract API version from request"""
		try:
			extractor = self.version_extractors.get(self.versioning_strategy)
			if not extractor:
				return self.default_version
			
			version = extractor(request_info)
			
			# Validate version exists
			if version and version in self.active_versions:
				return version
			
			# Return default version if extraction fails
			return self.default_version
			
		except Exception as e:
			logger.error(f"Failed to extract version from request: {str(e)}")
			return self.default_version

	async def transform_request_data(
		self,
		data: Dict[str, Any],
		from_version: str,
		to_version: str
	) -> Dict[str, Any]:
		"""Transform request data between versions"""
		try:
			if from_version == to_version:
				return data
			
			migration_key = f"{from_version}->{to_version}"
			migration = self.migration_paths.get(migration_key)
			
			if not migration:
				# Try to find indirect migration path
				migration = await self._find_migration_path(from_version, to_version)
			
			if not migration:
				logger.warning(f"No migration path found: {from_version} -> {to_version}")
				return data
			
			# Apply field mappings
			transformed_data = {}
			for key, value in data.items():
				new_key = migration.field_mappings.get(key, key)
				if new_key and key not in migration.deprecated_fields:
					transformed_data[new_key] = value
			
			# Apply data transformations
			for transformation in migration.data_transformations:
				transformed_data = await self._apply_transformation(transformed_data, transformation)
			
			return transformed_data
			
		except Exception as e:
			logger.error(f"Failed to transform request data: {str(e)}")
			return data  # Return original data on error

	async def get_deprecation_warnings(
		self,
		version_number: str,
		endpoint: Optional[str] = None
	) -> List[Dict[str, Any]]:
		"""Get deprecation warnings for version/endpoint"""
		try:
			warnings = []
			
			# Check version deprecation
			notice = self.deprecation_notices.get(version_number)
			if notice:
				if not endpoint or not notice.endpoint_pattern or endpoint.match(notice.endpoint_pattern):
					warnings.append({
						"type": "version_deprecation",
						"severity": notice.severity.value,
						"title": notice.title,
						"message": notice.message,
						"deprecation_date": notice.deprecation_date.isoformat(),
						"sunset_date": notice.sunset_date.isoformat() if notice.sunset_date else None,
						"replacement": notice.replacement_endpoint,
						"migration_guide": notice.migration_guide_url
					})
			
			return warnings
			
		except Exception as e:
			logger.error(f"Failed to get deprecation warnings: {str(e)}")
			return []

	async def track_client_usage(
		self,
		client_id: str,
		version_number: str,
		endpoint: str,
		response_time_ms: float,
		success: bool = True
	) -> None:
		"""Track client API usage"""
		try:
			# Update in-memory tracking
			if client_id not in self.client_usage:
				self.client_usage[client_id] = {}
			
			if version_number not in self.client_usage[client_id]:
				self.client_usage[client_id][version_number] = {
					"total_requests": 0,
					"successful_requests": 0,
					"failed_requests": 0,
					"avg_response_time": 0.0,
					"endpoints": {},
					"last_request_at": None
				}
			
			usage = self.client_usage[client_id][version_number]
			usage["total_requests"] += 1
			
			if success:
				usage["successful_requests"] += 1
			else:
				usage["failed_requests"] += 1
			
			# Update average response time
			current_avg = usage["avg_response_time"]
			total_requests = usage["total_requests"]
			usage["avg_response_time"] = ((current_avg * (total_requests - 1)) + response_time_ms) / total_requests
			
			usage["last_request_at"] = datetime.utcnow()
			
			# Track endpoint usage
			if endpoint not in usage["endpoints"]:
				usage["endpoints"][endpoint] = 0
			usage["endpoints"][endpoint] += 1
			
			# Periodically persist to database (every 100 requests)
			if usage["total_requests"] % 100 == 0:
				await self._persist_client_usage(client_id, version_number, usage)
				
		except Exception as e:
			logger.error(f"Failed to track client usage: {str(e)}")

	# Internal methods

	async def _load_active_versions(self) -> None:
		"""Load active API versions from database"""
		try:
			async with self.db_pool.acquire() as conn:
				rows = await conn.fetch("""
					SELECT * FROM crm_api_versions 
					WHERE is_active = true 
					ORDER BY created_at DESC
				""")
				
				for row in rows:
					version_data = dict(row)
					# Parse JSON fields
					version_data['changelog'] = json.loads(version_data['changelog'])
					version_data['breaking_changes'] = json.loads(version_data['breaking_changes'])
					version_data['new_features'] = json.loads(version_data['new_features'])
					version_data['bug_fixes'] = json.loads(version_data['bug_fixes'])
					version_data['supported_versions'] = json.loads(version_data['supported_versions'])
					version_data['migration_paths'] = json.loads(version_data['migration_paths'])
					version_data['api_endpoints'] = json.loads(version_data['api_endpoints'])
					version_data['supported_formats'] = json.loads(version_data['supported_formats'])
					version_data['authentication_methods'] = json.loads(version_data['authentication_methods'])
					version_data['usage_stats'] = json.loads(version_data['usage_stats'])
					version_data['client_adoption'] = json.loads(version_data['client_adoption'])
					
					version = APIVersion(**version_data)
					self.active_versions[version.version_number] = version
					
					if version.is_default:
						self.default_version = version.version_number
			
			# Set first version as default if none specified
			if not self.default_version and self.active_versions:
				first_version = next(iter(self.active_versions.keys()))
				self.default_version = first_version
			
			logger.info(f"Loaded {len(self.active_versions)} active API versions")
			
		except Exception as e:
			logger.error(f"Failed to load active versions: {str(e)}")

	async def _load_deprecation_notices(self) -> None:
		"""Load deprecation notices from database"""
		try:
			async with self.db_pool.acquire() as conn:
				rows = await conn.fetch("""
					SELECT dn.*, av.version_number 
					FROM crm_deprecation_notices dn
					JOIN crm_api_versions av ON dn.version_id = av.id
					WHERE av.is_active = true
				""")
				
				for row in rows:
					notice_data = dict(row)
					notice_data['announcement_channels'] = json.loads(notice_data['announcement_channels'])
					notice_data['affected_clients'] = json.loads(notice_data['affected_clients'])
					notice_data['client_acknowledgments'] = json.loads(notice_data['client_acknowledgments'])
					
					notice = DeprecationNotice(**notice_data)
					self.deprecation_notices[notice_data['version_number']] = notice
			
			logger.info(f"Loaded {len(self.deprecation_notices)} deprecation notices")
			
		except Exception as e:
			logger.error(f"Failed to load deprecation notices: {str(e)}")

	async def _load_migration_paths(self) -> None:
		"""Load migration paths from database"""
		try:
			async with self.db_pool.acquire() as conn:
				rows = await conn.fetch("SELECT * FROM crm_version_migrations")
				
				for row in rows:
					migration_data = dict(row)
					# Parse JSON fields
					migration_data['migration_steps'] = json.loads(migration_data['migration_steps'])
					migration_data['validation_rules'] = json.loads(migration_data['validation_rules'])
					migration_data['rollback_steps'] = json.loads(migration_data['rollback_steps'])
					migration_data['field_mappings'] = json.loads(migration_data['field_mappings'])
					migration_data['deprecated_fields'] = json.loads(migration_data['deprecated_fields'])
					migration_data['new_fields'] = json.loads(migration_data['new_fields'])
					migration_data['data_transformations'] = json.loads(migration_data['data_transformations'])
					migration_data['custom_transformers'] = json.loads(migration_data['custom_transformers'])
					migration_data['code_examples'] = json.loads(migration_data['code_examples'])
					
					migration = VersionMigration(**migration_data)
					migration_key = f"{migration.from_version}->{migration.to_version}"
					self.migration_paths[migration_key] = migration
			
			logger.info(f"Loaded {len(self.migration_paths)} migration paths")
			
		except Exception as e:
			logger.error(f"Failed to load migration paths: {str(e)}")

	async def _load_client_usage(self) -> None:
		"""Load client usage data from database"""
		try:
			async with self.db_pool.acquire() as conn:
				rows = await conn.fetch("""
					SELECT * FROM crm_client_version_usage 
					WHERE updated_at > NOW() - INTERVAL '7 days'
				""")
				
				for row in rows:
					usage_data = dict(row)
					usage_data['version_usage'] = json.loads(usage_data['version_usage'])
					usage_data['fallback_versions'] = json.loads(usage_data['fallback_versions'])
					usage_data['contact_info'] = json.loads(usage_data['contact_info'])
					usage_data['migration_status'] = json.loads(usage_data['migration_status'])
					usage_data['migration_deadlines'] = json.loads(usage_data['migration_deadlines'])
					
					client_usage = ClientVersionUsage(**usage_data)
					self.client_usage[client_usage.client_id] = client_usage.version_usage
			
			logger.info(f"Loaded usage data for {len(self.client_usage)} clients")
			
		except Exception as e:
			logger.error(f"Failed to load client usage: {str(e)}")

	def _extract_version_from_path(self, request_info: Dict[str, Any]) -> Optional[str]:
		"""Extract version from URL path"""
		path = request_info.get('path', '')
		# Match patterns like /v1/, /v2.1/, /api/v3/
		match = re.search(r'/v(\d+(?:\.\d+)?)', path)
		return match.group(1) if match else None

	def _extract_version_from_header(self, request_info: Dict[str, Any]) -> Optional[str]:
		"""Extract version from header"""
		headers = request_info.get('headers', {})
		return headers.get('X-API-Version') or headers.get('API-Version')

	def _extract_version_from_query(self, request_info: Dict[str, Any]) -> Optional[str]:
		"""Extract version from query parameter"""
		query_params = request_info.get('query_params', {})
		return query_params.get('version') or query_params.get('api_version')

	def _extract_version_from_content_type(self, request_info: Dict[str, Any]) -> Optional[str]:
		"""Extract version from content type"""
		headers = request_info.get('headers', {})
		content_type = headers.get('Content-Type', '')
		# Match patterns like application/vnd.api+json;version=1.0
		match = re.search(r'version=(\d+(?:\.\d+)?)', content_type)
		return match.group(1) if match else None

	async def _initialize_transformations(self) -> None:
		"""Initialize data transformation functions"""
		self.transformation_functions = {
			'rename_field': self._transform_rename_field,
			'convert_type': self._transform_convert_type,
			'split_field': self._transform_split_field,
			'merge_fields': self._transform_merge_fields,
			'apply_default': self._transform_apply_default,
			'validate_format': self._transform_validate_format
		}

	async def _start_deprecation_scheduler(self) -> None:
		"""Start background task for deprecation management"""
		async def deprecation_monitor():
			while True:
				try:
					await self._check_deprecation_deadlines()
					await asyncio.sleep(3600)  # Check hourly
				except Exception as e:
					logger.error(f"Deprecation monitor error: {str(e)}")
					await asyncio.sleep(3600)
		
		asyncio.create_task(deprecation_monitor())
		logger.info("Deprecation scheduler started")

	async def shutdown(self) -> None:
		"""Shutdown API versioning manager"""
		try:
			# Persist any pending client usage data
			for client_id, versions in self.client_usage.items():
				for version_number, usage in versions.items():
					await self._persist_client_usage(client_id, version_number, usage)
			
			logger.info("API versioning manager shutdown completed")
			
		except Exception as e:
			logger.error(f"Error during versioning manager shutdown: {str(e)}")

	# Helper transformation methods
	async def _transform_rename_field(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
		"""Rename field transformation"""
		old_name = config['old_name']
		new_name = config['new_name']
		
		if old_name in data:
			data[new_name] = data.pop(old_name)
		
		return data

	async def _transform_convert_type(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
		"""Type conversion transformation"""
		field_name = config['field']
		target_type = config['type']
		
		if field_name in data:
			value = data[field_name]
			if target_type == 'string':
				data[field_name] = str(value)
			elif target_type == 'integer':
				data[field_name] = int(value)
			elif target_type == 'float':
				data[field_name] = float(value)
			elif target_type == 'boolean':
				data[field_name] = bool(value)
		
		return data

	async def _persist_client_usage(self, client_id: str, version_number: str, usage: Dict[str, Any]) -> None:
		"""Persist client usage data to database"""
		try:
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_client_version_usage (
						id, client_id, version_number, total_requests,
						successful_requests, failed_requests, avg_response_time_ms,
						endpoint_usage, last_request_at, updated_at
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
					ON CONFLICT (client_id, version_number) 
					DO UPDATE SET
						total_requests = EXCLUDED.total_requests,
						successful_requests = EXCLUDED.successful_requests,
						failed_requests = EXCLUDED.failed_requests,
						avg_response_time_ms = EXCLUDED.avg_response_time_ms,
						endpoint_usage = EXCLUDED.endpoint_usage,
						last_request_at = EXCLUDED.last_request_at,
						updated_at = EXCLUDED.updated_at
				""",
				uuid7str(), client_id, version_number, usage["total_requests"],
				usage["successful_requests"], usage["failed_requests"],
				usage["avg_response_time"], json.dumps(usage["endpoints"]),
				usage["last_request_at"], datetime.utcnow())
				
		except Exception as e:
			logger.error(f"Failed to persist client usage: {str(e)}")