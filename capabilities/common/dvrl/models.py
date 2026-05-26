#!/usr/bin/env python3
"""
APG Data Virtualization (DVRL) Core Models
Data models for federated query processing and data source management

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from pydantic.functional_validators import BeforeValidator
from typing import Annotated


# APG Multi-Tenancy Support
class APGTenantModel(BaseModel):
	"""Base model with APG multi-tenancy support"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True
	)
	
	tenant_id: str = Field(..., description="APG tenant identifier")
	created_by: str = Field(..., description="APG user who created this entity")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
	updated_by: Optional[str] = Field(None, description="APG user who last updated this entity")
	updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# Data Source Types and Status
class DataSourceType(str, Enum):
	"""Supported data source types"""
	POSTGRESQL = "postgresql"
	MYSQL = "mysql"
	ORACLE = "oracle"
	SQLSERVER = "sqlserver"
	MONGODB = "mongodb"
	CASSANDRA = "cassandra"
	REDIS = "redis"
	ELASTICSEARCH = "elasticsearch"
	SNOWFLAKE = "snowflake"
	BIGQUERY = "bigquery"
	REDSHIFT = "redshift"
	S3 = "s3"
	HDFS = "hdfs"
	BYTEWAX = "bytewax"
	REST_API = "rest_api"
	GRAPHQL = "graphql"
	FILE_CSV = "file_csv"
	FILE_JSON = "file_json"
	FILE_PARQUET = "file_parquet"


class DataSourceStatus(str, Enum):
	"""Data source connection status"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	TESTING = "testing"
	ERROR = "error"
	MAINTENANCE = "maintenance"


class QueryStatus(str, Enum):
	"""Query execution status"""
	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	CACHED = "cached"


class CacheLevel(str, Enum):
	"""Cache hierarchy levels"""
	MEMORY = "memory"
	DISK = "disk"
	DISTRIBUTED = "distributed"
	PERSISTENT = "persistent"


# Core Data Models
class DataSource(APGTenantModel):
	"""Data source configuration and metadata"""
	
	id: str = Field(default_factory=uuid7str, description="Unique data source identifier")
	name: str = Field(..., min_length=1, max_length=255, description="Human-readable data source name")
	description: Optional[str] = Field(None, max_length=1000, description="Data source description")
	type: DataSourceType = Field(..., description="Type of data source")
	status: DataSourceStatus = Field(default=DataSourceStatus.ACTIVE, description="Connection status")
	
	# Connection Configuration
	connection_config: Dict[str, Any] = Field(..., description="Connection parameters and credentials")
	connection_string: Optional[str] = Field(None, description="Optional connection string")
	host: Optional[str] = Field(None, description="Host address")
	port: Optional[int] = Field(None, description="Port number")
	database: Optional[str] = Field(None, description="Database name")
	schema: Optional[str] = Field(None, description="Schema name")
	
	# Performance Configuration
	connection_pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
	query_timeout_seconds: int = Field(default=30, ge=1, le=3600, description="Query timeout")
	max_concurrent_queries: int = Field(default=5, ge=1, le=50, description="Max concurrent queries")
	
	# APG Integration
	metadata_schema_id: Optional[str] = Field(None, description="APG metadata schema reference")
	mdm_source_id: Optional[str] = Field(None, description="APG MDM source reference")
	audit_enabled: bool = Field(default=True, description="Enable audit logging")
	
	# Health and Monitoring
	health_check_enabled: bool = Field(default=True, description="Enable health monitoring")
	last_health_check: Optional[datetime] = Field(None, description="Last health check timestamp")
	is_healthy: bool = Field(default=True, description="Current health status")
	error_message: Optional[str] = Field(None, description="Last error message")
	
	# Usage Statistics
	query_count: int = Field(default=0, description="Total query count")
	last_accessed: Optional[datetime] = Field(None, description="Last access timestamp")
	avg_response_time_ms: Optional[float] = Field(None, description="Average response time")
	
	# Tags and Classification
	tags: List[str] = Field(default_factory=list, description="Data source tags")
	classification: Optional[str] = Field(None, description="Data classification level")
	owner_email: Optional[str] = Field(None, description="Data source owner")


class VirtualTable(APGTenantModel):
	"""Virtual table representing federated data source"""
	
	id: str = Field(default_factory=uuid7str, description="Unique virtual table identifier")
	name: str = Field(..., min_length=1, max_length=255, description="Virtual table name")
	description: Optional[str] = Field(None, max_length=1000, description="Table description")
	
	# Data Source Mapping
	data_source_id: str = Field(..., description="Associated data source ID")
	source_table: Optional[str] = Field(None, description="Source table/collection name")
	source_query: Optional[str] = Field(None, description="Custom source query")
	
	# Schema Definition
	columns: List[Dict[str, Any]] = Field(default_factory=list, description="Column definitions")
	primary_key: Optional[List[str]] = Field(None, description="Primary key columns")
	indexes: List[Dict[str, Any]] = Field(default_factory=list, description="Index definitions")
	
	# APG Metadata Integration
	metadata_table_id: Optional[str] = Field(None, description="APG metadata table reference")
	lineage_info: Dict[str, Any] = Field(default_factory=dict, description="Data lineage information")
	quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Data quality score")
	
	# Access Control
	access_policy: Dict[str, Any] = Field(default_factory=dict, description="Table access policies")
	row_level_security: bool = Field(default=False, description="Enable row-level security")
	column_level_security: bool = Field(default=False, description="Enable column-level security")
	
	# Performance Optimization
	partitioning_config: Optional[Dict[str, Any]] = Field(None, description="Partitioning configuration")
	indexing_strategy: Optional[Dict[str, Any]] = Field(None, description="Indexing strategy")
	cache_strategy: Optional[Dict[str, Any]] = Field(None, description="Caching strategy")


class FederatedQuery(APGTenantModel):
	"""Federated query execution metadata and results"""
	
	id: str = Field(default_factory=uuid7str, description="Unique query identifier")
	query_hash: str = Field(..., description="Hash of the query for caching")
	original_sql: str = Field(..., description="Original SQL query")
	optimized_sql: Optional[str] = Field(None, description="Optimized SQL query")
	
	# Query Classification
	query_type: str = Field(..., description="Query type (SELECT, INSERT, etc.)")
	complexity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Query complexity score")
	estimated_cost: Optional[float] = Field(None, description="Estimated execution cost")
	
	# Execution Details
	status: QueryStatus = Field(default=QueryStatus.PENDING, description="Query execution status")
	execution_plan: Optional[Dict[str, Any]] = Field(None, description="Query execution plan")
	data_sources_accessed: List[str] = Field(default_factory=list, description="Data sources accessed")
	tables_accessed: List[str] = Field(default_factory=list, description="Virtual tables accessed")
	
	# Performance Metrics
	started_at: Optional[datetime] = Field(None, description="Query start time")
	completed_at: Optional[datetime] = Field(None, description="Query completion time")
	duration_ms: Optional[int] = Field(None, description="Execution duration in milliseconds")
	rows_returned: Optional[int] = Field(None, description="Number of rows returned")
	bytes_processed: Optional[int] = Field(None, description="Bytes processed")
	
	# Caching Information
	cache_used: bool = Field(default=False, description="Whether cache was used")
	cache_level: Optional[CacheLevel] = Field(None, description="Cache level used")
	cache_hit_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="Cache hit ratio")
	
	# Error Information
	error_message: Optional[str] = Field(None, description="Error message if query failed")
	error_code: Optional[str] = Field(None, description="Error code")
	error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")
	
	# APG Integration
	audit_id: Optional[str] = Field(None, description="APG audit log reference")
	user_context: Dict[str, Any] = Field(default_factory=dict, description="User context information")
	security_policies_applied: List[str] = Field(default_factory=list, description="Applied security policies")


class QueryCache(APGTenantModel):
	"""Query result cache metadata and storage"""
	
	id: str = Field(default_factory=uuid7str, description="Unique cache entry identifier")
	query_hash: str = Field(..., description="Hash of the cached query")
	cache_key: str = Field(..., description="Cache storage key")
	
	# Cache Details
	cache_level: CacheLevel = Field(..., description="Cache storage level")
	result_size_bytes: int = Field(..., description="Size of cached result")
	row_count: int = Field(..., description="Number of cached rows")
	
	# Cache Management
	ttl_seconds: int = Field(default=3600, description="Time to live in seconds")
	expires_at: datetime = Field(..., description="Cache expiration time")
	access_count: int = Field(default=0, description="Number of times accessed")
	last_accessed: Optional[datetime] = Field(None, description="Last access time")
	
	# Invalidation Triggers
	invalidation_triggers: List[str] = Field(default_factory=list, description="Cache invalidation triggers")
	dependent_tables: List[str] = Field(default_factory=list, description="Dependent virtual tables")
	
	# Semantic Information for Smart Caching
	semantic_tags: List[str] = Field(default_factory=list, description="Semantic tags for similarity matching")
	embedding_vector: Optional[List[float]] = Field(None, description="Query embedding for similarity search")
	similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Similarity threshold for reuse")


class DataSourceSchema(APGTenantModel):
	"""Schema information for data sources"""
	
	id: str = Field(default_factory=uuid7str, description="Unique schema identifier")
	data_source_id: str = Field(..., description="Associated data source ID")
	schema_name: str = Field(..., description="Schema name")
	
	# Schema Details
	tables: List[Dict[str, Any]] = Field(default_factory=list, description="Table definitions")
	views: List[Dict[str, Any]] = Field(default_factory=list, description="View definitions")
	procedures: List[Dict[str, Any]] = Field(default_factory=list, description="Stored procedure definitions")
	functions: List[Dict[str, Any]] = Field(default_factory=list, description="Function definitions")
	
	# Schema Evolution
	version: str = Field(default="1.0.0", description="Schema version")
	previous_version: Optional[str] = Field(None, description="Previous schema version")
	evolution_history: List[Dict[str, Any]] = Field(default_factory=list, description="Schema change history")
	
	# Discovery Information
	discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Schema discovery time")
	discovery_method: str = Field(default="automatic", description="How schema was discovered")
	confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Discovery confidence")
	
	# APG Metadata Integration
	metadata_registered: bool = Field(default=False, description="Registered with APG metadata service")
	metadata_schema_id: Optional[str] = Field(None, description="APG metadata schema reference")


class FederationPlan(APGTenantModel):
	"""Query execution plan for federated queries"""
	
	id: str = Field(default_factory=uuid7str, description="Unique execution plan identifier")
	query_id: str = Field(..., description="Associated query ID")
	plan_hash: str = Field(..., description="Execution plan hash")
	
	# Plan Structure
	execution_steps: List[Dict[str, Any]] = Field(default_factory=list, description="Execution steps")
	data_movement_plan: Dict[str, Any] = Field(default_factory=dict, description="Data movement optimization")
	join_strategy: Dict[str, Any] = Field(default_factory=dict, description="Join execution strategy")
	
	# Cost Estimation
	estimated_cost: float = Field(..., description="Estimated execution cost")
	estimated_duration_ms: Optional[int] = Field(None, description="Estimated duration")
	estimated_memory_mb: Optional[float] = Field(None, description="Estimated memory usage")
	estimated_network_mb: Optional[float] = Field(None, description="Estimated network traffic")
	
	# Optimization Information
	optimization_level: str = Field(default="standard", description="Optimization level applied")
	optimization_techniques: List[str] = Field(default_factory=list, description="Optimization techniques used")
	alternative_plans: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative execution plans")
	
	# Execution Tracking
	actual_cost: Optional[float] = Field(None, description="Actual execution cost")
	actual_duration_ms: Optional[int] = Field(None, description="Actual execution duration")
	performance_ratio: Optional[float] = Field(None, description="Actual vs estimated performance")


# Validation Functions with AfterValidator
def validate_connection_config(v: Dict[str, Any]) -> Dict[str, Any]:
	"""Validate connection configuration"""
	if not v:
		raise ValueError("Connection configuration cannot be empty")
	
	# Remove sensitive information from logging
	safe_config = {k: "***" if "password" in k.lower() or "secret" in k.lower() or "key" in k.lower() else str(val)
					for k, val in v.items()}
	
	return v

def validate_query_sql(v: str) -> str:
	"""Basic SQL query validation"""
	if not v or not v.strip():
		raise ValueError("SQL query cannot be empty")
	
	# Basic SQL injection prevention
	dangerous_keywords = ['drop', 'delete', 'truncate', 'alter', 'create', 'exec']
	query_lower = v.lower().strip()
	
	for keyword in dangerous_keywords:
		if f' {keyword} ' in query_lower or query_lower.startswith(keyword):
			if not query_lower.startswith('create view') and not query_lower.startswith('create table'):
				raise ValueError(f"Potentially dangerous SQL keyword detected: {keyword}")
	
	return v.strip()

def validate_cache_ttl(v: int) -> int:
	"""Validate cache TTL"""
	if v < 60:  # Minimum 1 minute
		return 60
	if v > 86400 * 7:  # Maximum 1 week
		return 86400 * 7
	return v


# Apply validators to models
DataSource.model_fields['connection_config'] = Field(
	..., 
	description="Connection parameters and credentials",
	json_schema_extra={"validator": validate_connection_config}
)

FederatedQuery.model_fields['original_sql'] = Field(
	..., 
	description="Original SQL query",
	json_schema_extra={"validator": validate_query_sql}
)

QueryCache.model_fields['ttl_seconds'] = Field(
	default=3600, 
	description="Time to live in seconds",
	json_schema_extra={"validator": validate_cache_ttl}
)


# Helper Functions
def mask_sensitive_config(config: Dict[str, Any]) -> Dict[str, Any]:
	"""Mask sensitive information in configuration"""
	masked = {}
	for key, value in config.items():
		if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key', 'token']):
			masked[key] = "***"
		else:
			masked[key] = value
	return masked

async def calculate_query_complexity(sql: str) -> float:
	"""Calculate query complexity score"""
	# Simple complexity calculation based on SQL features
	complexity = 0.0
	sql_lower = sql.lower()
	
	# Base complexity
	complexity += 0.1
	
	# Joins increase complexity
	complexity += sql_lower.count('join') * 0.2
	
	# Subqueries increase complexity
	complexity += sql_lower.count('select') * 0.1 - 0.1  # Subtract base SELECT
	
	# Window functions increase complexity
	complexity += sql_lower.count('over(') * 0.3
	
	# Aggregations increase complexity
	for agg in ['group by', 'having', 'order by']:
		if agg in sql_lower:
			complexity += 0.2
	
	# CTEs increase complexity
	complexity += sql_lower.count('with ') * 0.3
	
	return min(complexity, 1.0)

async def estimate_query_cost(plan: Dict[str, Any]) -> float:
	"""Estimate query execution cost"""
	# Simple cost estimation based on plan structure
	base_cost = 1.0
	
	# Cost increases with number of data sources
	data_sources = len(plan.get('data_sources', []))
	source_cost = data_sources * 0.5
	
	# Cost increases with estimated rows
	estimated_rows = plan.get('estimated_rows', 1000)
	row_cost = (estimated_rows / 1000) * 0.1
	
	# Cost increases with joins
	join_count = len(plan.get('joins', []))
	join_cost = join_count * 0.3
	
	return base_cost + source_cost + row_cost + join_cost


# Model Export
__all__ = [
	"APGTenantModel",
	"DataSourceType",
	"DataSourceStatus", 
	"QueryStatus",
	"CacheLevel",
	"DataSource",
	"VirtualTable",
	"FederatedQuery",
	"QueryCache",
	"DataSourceSchema",
	"FederationPlan",
	"mask_sensitive_config",
	"calculate_query_complexity",
	"estimate_query_cost"
]
