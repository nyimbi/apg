#!/usr/bin/env python3
"""
APG Cache Management (CACH) - Data Models
Pydantic v2 models following APG coding standards

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
from uuid_extensions import uuid7str
import json
from dataclasses import dataclass


class CacheBackendType(str, Enum):
	"""Supported cache backend types"""
	REDIS = "redis"
	HAZELCAST = "hazelcast"
	MEMORY = "memory"
	EDGE = "edge"
	DISTRIBUTED = "distributed"


class CompressionAlgorithm(str, Enum):
	"""Supported compression algorithms"""
	NONE = "none"
	GZIP = "gzip"
	LZ4 = "lz4"
	ZSTD = "zstd"
	SNAPPY = "snappy"
	BROTLI = "brotli"


class EvictionPolicy(str, Enum):
	"""Cache eviction policies"""
	LRU = "lru"  # Least Recently Used
	LFU = "lfu"  # Least Frequently Used
	FIFO = "fifo"  # First In First Out
	TTL = "ttl"  # Time To Live
	ADAPTIVE = "adaptive"  # AI-driven adaptive eviction
	INTELLIGENT = "intelligent"  # ML-optimized eviction


class CacheAccessPattern(str, Enum):
	"""Cache access patterns for optimization"""
	READ_HEAVY = "read_heavy"
	WRITE_HEAVY = "write_heavy"
	MIXED = "mixed"
	TEMPORAL = "temporal"
	SPATIAL = "spatial"
	SEQUENTIAL = "sequential"
	RANDOM = "random"


class SecurityLevel(str, Enum):
	"""Security levels for cache encryption"""
	NONE = "none"
	BASIC = "basic"
	ENTERPRISE = "enterprise"
	QUANTUM_SAFE = "quantum_safe"


class CacheTier(str, Enum):
	"""Cache tier levels"""
	L1 = "l1"  # In-memory, fastest
	L2 = "l2"  # Redis-like, fast network
	L3 = "l3"  # Distributed, persistent
	EDGE = "edge"  # Edge locations, CDN-like


def _validate_positive_int(value: int) -> int:
	"""Validate that integer is positive"""
	if value <= 0:
		raise ValueError("Value must be positive")
	return value


def _validate_ttl_seconds(value: Optional[int]) -> Optional[int]:
	"""Validate TTL is reasonable"""
	if value is not None and (value < 1 or value > 31536000):  # 1 second to 1 year
		raise ValueError("TTL must be between 1 second and 1 year")
	return value


def _validate_size_bytes(value: int) -> int:
	"""Validate size is reasonable"""
	if value < 0 or value > 1073741824:  # Max 1GB per cache entry
		raise ValueError("Size must be between 0 and 1GB")
	return value


class CacheEntry(BaseModel):
	"""Individual cache entry with metadata and AI optimization data"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Core identification
	key: str = Field(..., description="Unique cache key identifier", max_length=1024)
	value: bytes = Field(..., description="Cached data value")
	
	# TTL and expiration
	ttl_seconds: Optional[int] = Field(
		None, 
		description="Time to live in seconds",
		validation_alias="ttl"
	)
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
	last_accessed: Optional[datetime] = Field(None, description="Last access timestamp")
	
	# Access tracking for AI optimization
	access_count: int = Field(default=0, description="Total access count", ge=0)
	access_frequency: float = Field(default=0.0, description="Access frequency (accesses/hour)", ge=0.0)
	hit_count: int = Field(default=0, description="Cache hit count", ge=0)
	miss_count: int = Field(default=0, description="Cache miss count", ge=0)
	
	# Size and compression
	size_bytes: int = Field(..., description="Size of cached value in bytes", ge=0)
	original_size_bytes: int = Field(..., description="Original uncompressed size", ge=0)
	compression_type: CompressionAlgorithm = Field(default=CompressionAlgorithm.NONE)
	compression_ratio: float = Field(default=1.0, description="Compression ratio", ge=0.0, le=1.0)
	
	# Multi-tenancy and security
	tenant_id: str = Field(..., description="APG tenant identifier")
	namespace: str = Field(default="default", description="Cache namespace")
	security_level: SecurityLevel = Field(default=SecurityLevel.BASIC)
	encrypted: bool = Field(default=True, description="Whether value is encrypted")
	
	# AI optimization metadata
	access_pattern: CacheAccessPattern = Field(default=CacheAccessPattern.MIXED)
	predicted_access_time: Optional[datetime] = Field(None, description="AI-predicted next access")
	optimization_score: float = Field(default=0.0, description="AI optimization score", ge=0.0, le=1.0)
	prefetch_candidate: bool = Field(default=False, description="Candidate for prefetching")
	tier_recommendation: CacheTier = Field(default=CacheTier.L1)
	
	# Content analysis
	content_hash: Optional[str] = Field(None, description="SHA-256 hash of content")
	content_type: Optional[str] = Field(None, description="MIME type or data type")
	semantic_tags: List[str] = Field(default_factory=list, description="AI-generated semantic tags")
	related_keys: List[str] = Field(default_factory=list, description="Related cache keys")
	
	# Performance metrics
	average_access_latency_ms: float = Field(default=0.0, description="Average access latency", ge=0.0)
	serialization_time_ms: float = Field(default=0.0, description="Serialization time", ge=0.0)
	network_transfer_time_ms: float = Field(default=0.0, description="Network transfer time", ge=0.0)
	
	def is_expired(self) -> bool:
		"""Check if cache entry has expired"""
		if self.expires_at:
			return datetime.utcnow() > self.expires_at
		if self.ttl_seconds and self.created_at:
			expiry = self.created_at + timedelta(seconds=self.ttl_seconds)
			return datetime.utcnow() > expiry
		return False
	
	def hit_rate(self) -> float:
		"""Calculate cache hit rate"""
		total = self.hit_count + self.miss_count
		return self.hit_count / max(1, total)
	
	def update_access_stats(self, hit: bool = True) -> None:
		"""Update access statistics"""
		self.access_count += 1
		self.last_accessed = datetime.utcnow()
		if hit:
			self.hit_count += 1
		else:
			self.miss_count += 1


class CacheCluster(BaseModel):
	"""Cache cluster configuration and management"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Core identification
	cluster_id: str = Field(default_factory=uuid7str, description="Unique cluster identifier")
	name: str = Field(..., description="Human-readable cluster name", max_length=255)
	description: str = Field("", description="Cluster description", max_length=1000)
	
	# Cluster configuration
	backend_type: CacheBackendType = Field(default=CacheBackendType.REDIS)
	nodes: List[str] = Field(default_factory=list, description="Node addresses (host:port)")
	replication_factor: int = Field(default=2, description="Data replication factor", ge=1, le=10)
	partitions: int = Field(default=16, description="Number of partitions", ge=1, le=1000)
	
	# Resource limits
	max_memory_mb: int = Field(
		default=1024, 
		description="Maximum memory per node in MB",
		validation_alias="max_memory"
	)
	max_connections: int = Field(default=1000, description="Max concurrent connections", ge=1)
	max_operations_per_second: int = Field(default=100000, description="Rate limit", ge=1)
	
	# Cache policies
	default_ttl_seconds: int = Field(default=3600, description="Default TTL in seconds", ge=1)
	eviction_policy: EvictionPolicy = Field(default=EvictionPolicy.ADAPTIVE)
	compression_enabled: bool = Field(default=True)
	default_compression: CompressionAlgorithm = Field(default=CompressionAlgorithm.LZ4)
	
	# Security configuration
	encryption_enabled: bool = Field(default=True)
	security_level: SecurityLevel = Field(default=SecurityLevel.ENTERPRISE)
	auth_required: bool = Field(default=True)
	tls_enabled: bool = Field(default=True)
	
	# Multi-tenancy
	tenant_id: str = Field(..., description="APG tenant identifier")
	tenant_isolation: bool = Field(default=True, description="Enable tenant isolation")
	resource_quotas: Dict[str, int] = Field(
		default_factory=dict,
		description="Per-tenant resource quotas"
	)
	
	# AI optimization settings
	ai_optimization_enabled: bool = Field(default=True)
	predictive_prefetching: bool = Field(default=True)
	auto_scaling: bool = Field(default=True)
	intelligent_eviction: bool = Field(default=True)
	performance_learning: bool = Field(default=True)
	
	# Monitoring and health
	health_check_interval_seconds: int = Field(default=30, ge=5)
	metrics_retention_days: int = Field(default=30, ge=1, le=365)
	alert_thresholds: Dict[str, float] = Field(default_factory=dict)
	
	# Metadata
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = Field(..., description="User who created the cluster")
	version: str = Field(default="1.0.0", description="Cluster configuration version")
	
	# Status tracking
	status: str = Field(default="initializing", description="Cluster status")
	last_health_check: Optional[datetime] = Field(None)
	healthy: bool = Field(default=True)
	
	def add_node(self, node_address: str) -> None:
		"""Add a node to the cluster"""
		if node_address not in self.nodes:
			self.nodes.append(node_address)
			self.updated_at = datetime.utcnow()
	
	def remove_node(self, node_address: str) -> None:
		"""Remove a node from the cluster"""
		if node_address in self.nodes:
			self.nodes.remove(node_address)
			self.updated_at = datetime.utcnow()
	
	def is_healthy(self) -> bool:
		"""Check if cluster is healthy"""
		if not self.healthy:
			return False
		if self.last_health_check:
			threshold = datetime.utcnow() - timedelta(seconds=self.health_check_interval_seconds * 2)
			return self.last_health_check > threshold
		return len(self.nodes) > 0


class CachePolicy(BaseModel):
	"""Cache policy configuration with AI-driven rules"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Core identification
	policy_id: str = Field(default_factory=uuid7str, description="Unique policy identifier")
	name: str = Field(..., description="Policy name", max_length=255)
	description: str = Field("", description="Policy description", max_length=1000)
	
	# Policy scope
	key_patterns: List[str] = Field(..., description="Key patterns this policy applies to")
	namespaces: List[str] = Field(default_factory=list, description="Target namespaces")
	tenant_id: str = Field(..., description="APG tenant identifier")
	
	# Basic cache settings
	default_ttl_seconds: int = Field(
		default=3600,
		description="Default TTL for matching keys",
		ge=1,
		le=31536000  # 1 year max
	)
	max_value_size_bytes: int = Field(
		default=1048576,  # 1MB
		description="Maximum value size",
		ge=1,
		le=1073741824  # 1GB max
	)
	
	# Optimization settings
	compression_enabled: bool = Field(default=True)
	compression_algorithm: CompressionAlgorithm = Field(default=CompressionAlgorithm.LZ4)
	compression_threshold_bytes: int = Field(default=1024, description="Min size for compression", ge=0)
	
	# AI-driven features
	prefetch_enabled: bool = Field(default=True, description="Enable intelligent prefetching")
	ai_optimization_enabled: bool = Field(default=True)
	adaptive_ttl: bool = Field(default=True, description="AI-adjusted TTL based on access patterns")
	intelligent_eviction: bool = Field(default=True)
	predictive_warming: bool = Field(default=True)
	
	# Performance optimization
	tier_preferences: List[CacheTier] = Field(
		default_factory=lambda: [CacheTier.L1, CacheTier.L2, CacheTier.L3],
		description="Preferred cache tiers in order"
	)
	replication_factor: int = Field(default=2, description="Data replication factor", ge=1, le=5)
	consistency_level: str = Field(default="eventual", description="Consistency requirements")
	
	# Security settings
	encryption_required: bool = Field(default=True)
	security_level: SecurityLevel = Field(default=SecurityLevel.ENTERPRISE)
	access_logging: bool = Field(default=True)
	audit_enabled: bool = Field(default=True)
	
	# Behavioral settings
	access_pattern_learning: bool = Field(default=True)
	usage_analytics: bool = Field(default=True)
	performance_monitoring: bool = Field(default=True)
	anomaly_detection: bool = Field(default=True)
	
	# Thresholds and limits
	max_memory_usage_percent: float = Field(default=80.0, ge=10.0, le=95.0)
	eviction_threshold_percent: float = Field(default=85.0, ge=50.0, le=95.0)
	prefetch_threshold_score: float = Field(default=0.7, ge=0.0, le=1.0)
	
	# Metadata
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = Field(..., description="User who created the policy")
	version: str = Field(default="1.0.0")
	enabled: bool = Field(default=True)
	
	# Policy effectiveness metrics
	applied_count: int = Field(default=0, description="Times policy was applied", ge=0)
	effectiveness_score: float = Field(default=0.0, description="Policy effectiveness", ge=0.0, le=1.0)
	performance_impact: float = Field(default=0.0, description="Performance impact score")
	
	def matches_key(self, key: str) -> bool:
		"""Check if policy applies to a given key"""
		import fnmatch
		for pattern in self.key_patterns:
			if fnmatch.fnmatch(key, pattern):
				return True
		return False
	
	def update_effectiveness(self, success: bool, performance_delta: float) -> None:
		"""Update policy effectiveness metrics"""
		self.applied_count += 1
		if success:
			self.effectiveness_score = (self.effectiveness_score * 0.9) + (1.0 * 0.1)
		else:
			self.effectiveness_score = (self.effectiveness_score * 0.9) + (0.0 * 0.1)
		self.performance_impact = (self.performance_impact * 0.9) + (performance_delta * 0.1)
		self.updated_at = datetime.utcnow()


class CacheMetrics(BaseModel):
	"""Cache performance and usage metrics"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Core identification
	metric_id: str = Field(default_factory=uuid7str)
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	tenant_id: str = Field(..., description="APG tenant identifier")
	cluster_id: Optional[str] = Field(None)
	
	# Performance metrics
	total_operations: int = Field(default=0, ge=0)
	cache_hits: int = Field(default=0, ge=0)
	cache_misses: int = Field(default=0, ge=0)
	cache_evictions: int = Field(default=0, ge=0)
	
	# Latency metrics (in milliseconds)
	average_latency_ms: float = Field(default=0.0, ge=0.0)
	p50_latency_ms: float = Field(default=0.0, ge=0.0)
	p95_latency_ms: float = Field(default=0.0, ge=0.0)
	p99_latency_ms: float = Field(default=0.0, ge=0.0)
	
	# Throughput metrics
	operations_per_second: float = Field(default=0.0, ge=0.0)
	bytes_per_second: float = Field(default=0.0, ge=0.0)
	requests_per_minute: float = Field(default=0.0, ge=0.0)
	
	# Memory metrics
	total_memory_bytes: int = Field(default=0, ge=0)
	used_memory_bytes: int = Field(default=0, ge=0)
	available_memory_bytes: int = Field(default=0, ge=0)
	memory_utilization_percent: float = Field(default=0.0, ge=0.0, le=100.0)
	
	# AI optimization metrics
	ai_recommendations_generated: int = Field(default=0, ge=0)
	ai_optimizations_applied: int = Field(default=0, ge=0)
	prefetch_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
	prediction_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
	
	# Error metrics
	error_count: int = Field(default=0, ge=0)
	timeout_count: int = Field(default=0, ge=0)
	connection_errors: int = Field(default=0, ge=0)
	
	def hit_rate(self) -> float:
		"""Calculate cache hit rate"""
		total = self.cache_hits + self.cache_misses
		return self.cache_hits / max(1, total)
	
	def error_rate(self) -> float:
		"""Calculate error rate"""
		return self.error_count / max(1, self.total_operations)


class AIOptimizationResult(BaseModel):
	"""Results from AI optimization analysis"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Core identification
	result_id: str = Field(default_factory=uuid7str)
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	tenant_id: str = Field(..., description="APG tenant identifier")
	
	# Optimization target
	target_type: str = Field(..., description="What was optimized (cache_size, eviction_policy, etc.)")
	target_id: str = Field(..., description="ID of the optimized resource")
	
	# Recommendations
	recommendations: List[Dict[str, Any]] = Field(default_factory=list)
	confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendations")
	expected_improvement: float = Field(default=0.0, description="Expected performance improvement")
	
	# Analysis results
	current_performance: Dict[str, float] = Field(default_factory=dict)
	predicted_performance: Dict[str, float] = Field(default_factory=dict)
	optimization_factors: Dict[str, float] = Field(default_factory=dict)
	
	# Implementation status
	applied: bool = Field(default=False)
	applied_at: Optional[datetime] = Field(None)
	actual_improvement: Optional[float] = Field(None)
	
	# Metadata
	model_version: str = Field(default="1.0.0")
	analysis_duration_ms: float = Field(default=0.0, ge=0.0)


# Export all models
__all__ = [
	'CacheBackendType', 'CompressionAlgorithm', 'EvictionPolicy', 'CacheAccessPattern',
	'SecurityLevel', 'CacheTier', 'CacheEntry', 'CacheCluster', 'CachePolicy', 
	'CacheMetrics', 'AIOptimizationResult'
]