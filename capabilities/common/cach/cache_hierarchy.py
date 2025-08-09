#!/usr/bin/env python3
"""
APG Cache Management (CACH) - Cache Hierarchy Management
Intelligent multi-tier orchestration with dynamic optimization

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import math
import hashlib

from .models import CacheEntry, CacheCluster, CacheAccessPattern, CacheTier, SecurityLevel
from .ai_optimization import OptimizationMetrics


class TierStrategy(str, Enum):
	"""Cache tier placement strategies"""
	FREQUENCY_BASED = "frequency_based"
	SIZE_BASED = "size_based"
	LATENCY_OPTIMIZED = "latency_optimized"
	COST_OPTIMIZED = "cost_optimized"
	AI_OPTIMIZED = "ai_optimized"
	HYBRID = "hybrid"


class ConsistencyLevel(str, Enum):
	"""Cache consistency levels across tiers"""
	EVENTUAL = "eventual"
	STRONG = "strong"
	SESSION = "session"
	MONOTONIC_READ = "monotonic_read"
	CAUSAL = "causal"


@dataclass
class TierConfiguration:
	"""Configuration for a cache tier"""
	tier: CacheTier
	capacity_mb: int
	max_entry_size_mb: int
	target_latency_ms: float
	consistency_level: ConsistencyLevel
	replication_factor: int
	eviction_threshold: float  # 0.0 to 1.0
	backends: List[str]  # Backend node addresses
	encryption_enabled: bool = True
	compression_enabled: bool = True
	
	# Performance characteristics
	estimated_latency_ms: float = 0.0
	throughput_ops_per_sec: int = 0
	cost_per_gb_per_hour: float = 0.0


@dataclass
class TierMetrics:
	"""Performance metrics for a cache tier"""
	tier: CacheTier
	current_size_mb: float = 0.0
	utilization_percent: float = 0.0
	hit_rate: float = 0.0
	average_latency_ms: float = 0.0
	operations_per_second: float = 0.0
	evictions_per_hour: float = 0.0
	error_rate: float = 0.0
	last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataPlacementDecision:
	"""Decision about data placement across tiers"""
	key: str
	recommended_tier: CacheTier
	current_tier: Optional[CacheTier]
	confidence_score: float
	reasoning: str
	expected_performance_gain: float
	migration_cost: float
	alternative_tiers: List[Tuple[CacheTier, float]]  # (tier, score)


class MultiTierCacheHierarchy:
	"""
	Revolutionary multi-tier cache hierarchy with intelligent orchestration
	Revolutionary Differentiator #4: Adaptive Multi-Tier Orchestration
	"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		self.logger = logging.getLogger('cach.hierarchy')
		
		# Tier configurations
		self.tier_configs: Dict[CacheTier, TierConfiguration] = {}
		self.tier_metrics: Dict[CacheTier, TierMetrics] = {}
		self.tier_backends: Dict[CacheTier, List[Any]] = {}
		
		# Data placement state
		self.data_placement: Dict[str, CacheTier] = {}  # key -> current tier
		self.placement_history: deque = deque(maxlen=10000)
		self.migration_queue: deque = deque()
		
		# Optimization state
		self.placement_strategy = TierStrategy.AI_OPTIMIZED
		self.optimization_scores: Dict[str, float] = {}
		self.tier_load_balancing: Dict[CacheTier, float] = {}
		
		# Configuration parameters
		self.optimization_interval = 300  # 5 minutes
		self.migration_batch_size = 10
		self.consistency_timeout_ms = 5000
		self.tier_health_check_interval = 60  # 1 minute
		
		# Performance tracking
		self.cross_tier_operations = 0
		self.tier_hit_rates: Dict[CacheTier, deque] = defaultdict(lambda: deque(maxlen=100))
		self.placement_effectiveness: deque = deque(maxlen=1000)
	
	async def initialize(self) -> None:
		"""Initialize multi-tier cache hierarchy"""
		self.logger.info("Initializing multi-tier cache hierarchy...")
		
		# Setup default tier configurations
		await self._setup_default_tiers()
		
		# Initialize tier backends
		await self._initialize_tier_backends()
		
		# Start background optimization
		await self._start_tier_optimization()
		
		self.logger.info("Multi-tier cache hierarchy initialized")
	
	async def shutdown(self) -> None:
		"""Shutdown hierarchy gracefully"""
		self.logger.info("Shutting down multi-tier cache hierarchy...")
		
		# Stop background tasks
		# Save placement state
		
		self.logger.info("Multi-tier cache hierarchy shut down")
	
	async def place_data_optimally(self, entry: CacheEntry) -> DataPlacementDecision:
		"""
		Determine optimal tier placement for cache entry
		Revolutionary Differentiator #4: Dynamic Cache Hierarchy Management
		"""
		
		# Analyze entry characteristics
		entry_analysis = await self._analyze_entry_characteristics(entry)
		
		# Generate placement options
		placement_options = await self._generate_placement_options(entry, entry_analysis)
		
		# Score placement options
		scored_options = []
		for tier, base_score in placement_options:
			final_score = await self._calculate_placement_score(entry, tier, entry_analysis)
			scored_options.append((tier, final_score))
		
		# Select best option
		scored_options.sort(key=lambda x: x[1], reverse=True)
		best_tier, best_score = scored_options[0]
		
		# Create placement decision
		current_tier = self.data_placement.get(entry.key)
		decision = DataPlacementDecision(
			key=entry.key,
			recommended_tier=best_tier,
			current_tier=current_tier,
			confidence_score=best_score,
			reasoning=self._generate_placement_reasoning(entry, best_tier, entry_analysis),
			expected_performance_gain=await self._calculate_performance_gain(entry, current_tier, best_tier),
			migration_cost=await self._calculate_migration_cost(entry, current_tier, best_tier),
			alternative_tiers=scored_options[1:4]  # Top 3 alternatives
		)
		
		self.logger.debug(f"Optimal placement for {entry.key}: {best_tier.value} (score: {best_score:.3f})")
		return decision
	
	async def migrate_data(self, key: str, source_tier: CacheTier, target_tier: CacheTier,
						   entry: CacheEntry) -> Dict[str, Any]:
		"""
		Migrate data between tiers with consistency guarantees
		Smart data placement algorithms
		"""
		
		migration_start = datetime.utcnow()
		migration_result = {
			'success': False,
			'key': key,
			'source_tier': source_tier.value,
			'target_tier': target_tier.value,
			'duration_ms': 0.0,
			'bytes_transferred': 0
		}
		
		try:
			# Validate migration
			if not await self._validate_migration(key, source_tier, target_tier):
				migration_result['error'] = "Migration validation failed"
				return migration_result
			
			# Get tier backends
			source_backend = self.tier_backends.get(source_tier)
			target_backend = self.tier_backends.get(target_tier)
			
			if not source_backend or not target_backend:
				migration_result['error'] = "Backend not available"
				return migration_result
			
			# Perform migration with consistency
			migration_result['bytes_transferred'] = entry.size_bytes
			
			# Write to target tier first (write-ahead)
			await self._write_to_tier(target_tier, key, entry)
			
			# Verify write success
			if not await self._verify_tier_write(target_tier, key, entry):
				migration_result['error'] = "Target write verification failed"
				return migration_result
			
			# Update placement tracking
			self.data_placement[key] = target_tier
			
			# Remove from source tier (after successful target write)
			await self._remove_from_tier(source_tier, key)
			
			# Update metrics
			await self._update_tier_metrics_after_migration(source_tier, target_tier, entry)
			
			migration_result['success'] = True
			
			self.logger.debug(f"Successfully migrated {key} from {source_tier.value} to {target_tier.value}")
			
		except Exception as e:
			self.logger.error(f"Migration failed for {key}: {e}")
			migration_result['error'] = str(e)
		
		finally:
			migration_result['duration_ms'] = (datetime.utcnow() - migration_start).total_seconds() * 1000
		
		return migration_result
	
	async def optimize_tier_placement(self, entries: Dict[str, CacheEntry]) -> Dict[str, Any]:
		"""
		Comprehensive tier placement optimization
		Automatic tier optimization based on access patterns
		"""
		
		optimization_start = datetime.utcnow()
		optimization_results = {
			'entries_analyzed': len(entries),
			'placement_decisions': 0,
			'migrations_recommended': 0,
			'migrations_executed': 0,
			'performance_improvement': 0.0,
			'decisions': []
		}
		
		# Analyze all entries for optimal placement
		placement_decisions = []
		for key, entry in entries.items():
			decision = await self.place_data_optimally(entry)
			placement_decisions.append(decision)
			optimization_results['decisions'].append({
				'key': key,
				'current_tier': decision.current_tier.value if decision.current_tier else None,
				'recommended_tier': decision.recommended_tier.value,
				'confidence': decision.confidence_score,
				'reasoning': decision.reasoning
			})
		
		optimization_results['placement_decisions'] = len(placement_decisions)
		
		# Filter for high-confidence migrations
		migration_candidates = [
			decision for decision in placement_decisions
			if (decision.current_tier != decision.recommended_tier and
				decision.confidence_score > 0.7 and
				decision.expected_performance_gain > decision.migration_cost * 2)
		]
		
		optimization_results['migrations_recommended'] = len(migration_candidates)
		
		# Execute migrations (batched)
		migrations_executed = 0
		total_performance_gain = 0.0
		
		for decision in migration_candidates[:self.migration_batch_size]:
			if decision.key in entries:
				entry = entries[decision.key]
				if decision.current_tier:
					migration_result = await self.migrate_data(
						decision.key, decision.current_tier, decision.recommended_tier, entry
					)
					
					if migration_result['success']:
						migrations_executed += 1
						total_performance_gain += decision.expected_performance_gain
		
		optimization_results['migrations_executed'] = migrations_executed
		optimization_results['performance_improvement'] = total_performance_gain
		
		# Update tier load balancing
		await self._rebalance_tier_loads()
		
		# Record optimization effectiveness
		effectiveness = migrations_executed / max(len(migration_candidates), 1)
		self.placement_effectiveness.append(effectiveness)
		
		optimization_duration = (datetime.utcnow() - optimization_start).total_seconds() * 1000
		self.logger.info(f"Tier optimization completed in {optimization_duration:.1f}ms: "
						f"{migrations_executed}/{len(migration_candidates)} migrations executed")
		
		return optimization_results
	
	async def get_tier_recommendations(self, key: str, entry: CacheEntry) -> List[Dict[str, Any]]:
		"""Get tier placement recommendations with detailed analysis"""
		
		decision = await self.place_data_optimally(entry)
		
		recommendations = []
		for tier, score in [
			(decision.recommended_tier, decision.confidence_score)
		] + decision.alternative_tiers:
			
			tier_config = self.tier_configs.get(tier)
			tier_metrics = self.tier_metrics.get(tier)
			
			if tier_config and tier_metrics:
				recommendations.append({
					'tier': tier.value,
					'score': score,
					'estimated_latency_ms': tier_config.estimated_latency_ms,
					'capacity_utilization': tier_metrics.utilization_percent,
					'hit_rate': tier_metrics.hit_rate,
					'cost_per_hour': tier_config.cost_per_gb_per_hour * (entry.size_bytes / (1024**3)),
					'reasoning': self._generate_tier_reasoning(entry, tier)
				})
		
		return recommendations
	
	async def ensure_consistency_across_tiers(self, key: str, consistency_level: ConsistencyLevel) -> bool:
		"""
		Ensure data consistency across cache tiers
		Smart consistency management
		"""
		
		if consistency_level == ConsistencyLevel.EVENTUAL:
			return True  # No immediate consistency required
		
		tiers_with_key = []
		for tier in CacheTier:
			if await self._key_exists_in_tier(tier, key):
				tiers_with_key.append(tier)
		
		if len(tiers_with_key) <= 1:
			return True  # Only one copy, consistency guaranteed
		
		# Implement consistency protocol based on level
		if consistency_level == ConsistencyLevel.STRONG:
			return await self._ensure_strong_consistency(key, tiers_with_key)
		elif consistency_level == ConsistencyLevel.SESSION:
			return await self._ensure_session_consistency(key, tiers_with_key)
		elif consistency_level == ConsistencyLevel.MONOTONIC_READ:
			return await self._ensure_monotonic_read_consistency(key, tiers_with_key)
		elif consistency_level == ConsistencyLevel.CAUSAL:
			return await self._ensure_causal_consistency(key, tiers_with_key)
		
		return False
	
	async def get_hierarchy_statistics(self) -> Dict[str, Any]:
		"""Get comprehensive hierarchy statistics"""
		
		stats = {
			'tiers': {},
			'total_entries': len(self.data_placement),
			'cross_tier_operations': self.cross_tier_operations,
			'optimization_effectiveness': sum(self.placement_effectiveness) / max(len(self.placement_effectiveness), 1),
			'migration_queue_size': len(self.migration_queue)
		}
		
		for tier in CacheTier:
			tier_metrics = self.tier_metrics.get(tier)
			tier_config = self.tier_configs.get(tier)
			
			if tier_metrics and tier_config:
				tier_entries = sum(1 for t in self.data_placement.values() if t == tier)
				
				stats['tiers'][tier.value] = {
					'entries': tier_entries,
					'size_mb': tier_metrics.current_size_mb,
					'capacity_mb': tier_config.capacity_mb,
					'utilization_percent': tier_metrics.utilization_percent,
					'hit_rate': tier_metrics.hit_rate,
					'latency_ms': tier_metrics.average_latency_ms,
					'ops_per_second': tier_metrics.operations_per_second,
					'error_rate': tier_metrics.error_rate
				}
		
		return stats
	
	# Private implementation methods
	
	async def _setup_default_tiers(self) -> None:
		"""Setup default tier configurations"""
		
		# L1 Tier: In-memory, ultra-fast
		self.tier_configs[CacheTier.L1] = TierConfiguration(
			tier=CacheTier.L1,
			capacity_mb=1024,  # 1GB
			max_entry_size_mb=10,
			target_latency_ms=0.1,
			consistency_level=ConsistencyLevel.STRONG,
			replication_factor=1,
			eviction_threshold=0.9,
			backends=['memory://local'],
			estimated_latency_ms=0.1,
			throughput_ops_per_sec=1000000,
			cost_per_gb_per_hour=0.10
		)
		
		# L2 Tier: Redis-like, fast network cache
		self.tier_configs[CacheTier.L2] = TierConfiguration(
			tier=CacheTier.L2,
			capacity_mb=8192,  # 8GB
			max_entry_size_mb=100,
			target_latency_ms=1.0,
			consistency_level=ConsistencyLevel.SESSION,
			replication_factor=2,
			eviction_threshold=0.85,
			backends=['redis://cache-l2-1:6379', 'redis://cache-l2-2:6379'],
			estimated_latency_ms=1.0,
			throughput_ops_per_sec=100000,
			cost_per_gb_per_hour=0.05
		)
		
		# L3 Tier: Distributed, persistent
		self.tier_configs[CacheTier.L3] = TierConfiguration(
			tier=CacheTier.L3,
			capacity_mb=65536,  # 64GB
			max_entry_size_mb=1000,
			target_latency_ms=10.0,
			consistency_level=ConsistencyLevel.EVENTUAL,
			replication_factor=3,
			eviction_threshold=0.8,
			backends=['hazelcast://cache-l3-cluster'],
			estimated_latency_ms=10.0,
			throughput_ops_per_sec=10000,
			cost_per_gb_per_hour=0.02
		)
		
		# Edge Tier: Geographically distributed
		self.tier_configs[CacheTier.EDGE] = TierConfiguration(
			tier=CacheTier.EDGE,
			capacity_mb=4096,  # 4GB per edge location
			max_entry_size_mb=50,
			target_latency_ms=5.0,
			consistency_level=ConsistencyLevel.EVENTUAL,
			replication_factor=2,
			eviction_threshold=0.8,
			backends=['edge://us-east', 'edge://us-west', 'edge://eu-west'],
			estimated_latency_ms=5.0,
			throughput_ops_per_sec=50000,
			cost_per_gb_per_hour=0.08
		)
		
		# Initialize tier metrics
		for tier in CacheTier:
			self.tier_metrics[tier] = TierMetrics(tier=tier)
	
	async def _initialize_tier_backends(self) -> None:
		"""Initialize backends for each tier"""
		
		# Placeholder implementation - would initialize actual backends
		for tier in CacheTier:
			self.tier_backends[tier] = ["mock_backend"]
		
		self.logger.debug("Initialized tier backends")
	
	async def _start_tier_optimization(self) -> None:
		"""Start background tier optimization tasks"""
		
		# Would start actual background tasks
		self.logger.debug("Started tier optimization tasks")
	
	async def _analyze_entry_characteristics(self, entry: CacheEntry) -> Dict[str, Any]:
		"""Analyze cache entry characteristics for placement decisions"""
		
		characteristics = {
			'size_bytes': entry.size_bytes,
			'access_frequency': entry.access_frequency,
			'hit_rate': entry.hit_rate(),
			'access_pattern': entry.access_pattern,
			'last_accessed': entry.last_accessed,
			'compression_ratio': entry.compression_ratio,
			'ttl_seconds': entry.ttl_seconds,
			'prefetch_candidate': entry.prefetch_candidate
		}
		
		# Calculate derived metrics
		characteristics['size_category'] = self._categorize_by_size(entry.size_bytes)
		characteristics['frequency_category'] = self._categorize_by_frequency(entry.access_frequency)
		characteristics['recency_score'] = self._calculate_recency_score(entry.last_accessed)
		characteristics['value_density'] = entry.hit_rate() / max(entry.size_bytes / 1024, 1)  # hits per KB
		
		return characteristics
	
	async def _generate_placement_options(self, entry: CacheEntry, analysis: Dict[str, Any]) -> List[Tuple[CacheTier, float]]:
		"""Generate possible tier placement options with base scores"""
		
		options = []
		
		for tier in CacheTier:
			tier_config = self.tier_configs.get(tier)
			if not tier_config:
				continue
			
			# Check if entry fits in tier
			entry_size_mb = entry.size_bytes / (1024 * 1024)
			if entry_size_mb > tier_config.max_entry_size_mb:
				continue
			
			# Calculate base compatibility score
			base_score = 0.5  # Neutral starting point
			
			# Size compatibility
			if entry_size_mb <= tier_config.max_entry_size_mb * 0.1:  # Small items
				if tier == CacheTier.L1:
					base_score += 0.2
			elif entry_size_mb <= tier_config.max_entry_size_mb * 0.5:  # Medium items
				if tier in [CacheTier.L2, CacheTier.EDGE]:
					base_score += 0.2
			else:  # Large items
				if tier == CacheTier.L3:
					base_score += 0.2
			
			# Access pattern compatibility
			if analysis['frequency_category'] == 'high':
				if tier in [CacheTier.L1, CacheTier.L2]:
					base_score += 0.3
			elif analysis['frequency_category'] == 'low':
				if tier in [CacheTier.L3, CacheTier.EDGE]:
					base_score += 0.2
			
			options.append((tier, base_score))
		
		return options
	
	async def _calculate_placement_score(self, entry: CacheEntry, tier: CacheTier, 
										 analysis: Dict[str, Any]) -> float:
		"""Calculate comprehensive placement score for entry in specific tier"""
		
		tier_config = self.tier_configs.get(tier)
		tier_metrics = self.tier_metrics.get(tier)
		
		if not tier_config or not tier_metrics:
			return 0.0
		
		score = 0.0
		
		# Performance score (40% weight)
		performance_score = await self._calculate_performance_score(entry, tier, tier_config, analysis)
		score += performance_score * 0.4
		
		# Capacity utilization score (20% weight)
		capacity_score = self._calculate_capacity_score(entry, tier_config, tier_metrics)
		score += capacity_score * 0.2
		
		# Cost efficiency score (15% weight)
		cost_score = self._calculate_cost_score(entry, tier_config, analysis)
		score += cost_score * 0.15
		
		# Access pattern alignment (15% weight)
		pattern_score = self._calculate_pattern_alignment_score(entry, tier, analysis)
		score += pattern_score * 0.15
		
		# Tier load balancing (10% weight)
		load_score = self._calculate_load_balancing_score(tier)
		score += load_score * 0.1
		
		return min(score, 1.0)
	
	async def _calculate_performance_score(self, entry: CacheEntry, tier: CacheTier,
										   tier_config: TierConfiguration, analysis: Dict[str, Any]) -> float:
		"""Calculate performance score for placing entry in tier"""
		
		# Latency alignment
		if entry.access_frequency > 100:  # High frequency needs low latency
			if tier_config.estimated_latency_ms <= 1.0:
				latency_score = 1.0
			elif tier_config.estimated_latency_ms <= 5.0:
				latency_score = 0.7
			else:
				latency_score = 0.3
		else:  # Lower frequency can tolerate higher latency
			if tier_config.estimated_latency_ms <= 10.0:
				latency_score = 0.8
			else:
				latency_score = 0.6
		
		# Throughput alignment
		if entry.access_frequency > 50:
			throughput_score = min(tier_config.throughput_ops_per_sec / 100000, 1.0)
		else:
			throughput_score = 0.8  # Less critical for low-frequency items
		
		return (latency_score + throughput_score) / 2.0
	
	def _calculate_capacity_score(self, entry: CacheEntry, tier_config: TierConfiguration,
								  tier_metrics: TierMetrics) -> float:
		"""Calculate capacity utilization score"""
		
		current_utilization = tier_metrics.utilization_percent / 100.0
		entry_impact = (entry.size_bytes / (1024**2)) / tier_config.capacity_mb
		
		# Prefer tiers with good capacity headroom
		projected_utilization = current_utilization + entry_impact
		
		if projected_utilization <= 0.7:
			return 1.0
		elif projected_utilization <= 0.8:
			return 0.8
		elif projected_utilization <= 0.9:
			return 0.5
		else:
			return 0.1  # Avoid overloaded tiers
	
	def _calculate_cost_score(self, entry: CacheEntry, tier_config: TierConfiguration,
							  analysis: Dict[str, Any]) -> float:
		"""Calculate cost efficiency score"""
		
		# Calculate cost per access
		entry_size_gb = entry.size_bytes / (1024**3)
		hourly_cost = entry_size_gb * tier_config.cost_per_gb_per_hour
		
		# Factor in access frequency
		if entry.access_frequency > 0:
			cost_per_access = hourly_cost / entry.access_frequency
		else:
			cost_per_access = hourly_cost
		
		# Lower cost per access = higher score
		if cost_per_access <= 0.001:
			return 1.0
		elif cost_per_access <= 0.01:
			return 0.8
		elif cost_per_access <= 0.1:
			return 0.6
		else:
			return 0.3
	
	def _calculate_pattern_alignment_score(self, entry: CacheEntry, tier: CacheTier,
										   analysis: Dict[str, Any]) -> float:
		"""Calculate access pattern alignment score"""
		
		score = 0.5  # Base score
		
		# Frequency alignment
		if analysis['frequency_category'] == 'high' and tier in [CacheTier.L1, CacheTier.L2]:
			score += 0.3
		elif analysis['frequency_category'] == 'medium' and tier in [CacheTier.L2, CacheTier.L3]:
			score += 0.2
		elif analysis['frequency_category'] == 'low' and tier in [CacheTier.L3, CacheTier.EDGE]:
			score += 0.2
		
		# Size alignment
		if analysis['size_category'] == 'small' and tier == CacheTier.L1:
			score += 0.2
		elif analysis['size_category'] == 'large' and tier == CacheTier.L3:
			score += 0.2
		
		return min(score, 1.0)
	
	def _calculate_load_balancing_score(self, tier: CacheTier) -> float:
		"""Calculate load balancing score to distribute load evenly"""
		
		# Simple load balancing based on current utilization
		tier_metrics = self.tier_metrics.get(tier)
		if not tier_metrics:
			return 0.5
		
		utilization = tier_metrics.utilization_percent / 100.0
		
		# Prefer less utilized tiers
		return max(0.0, 1.0 - utilization)
	
	def _categorize_by_size(self, size_bytes: int) -> str:
		"""Categorize entry by size"""
		size_kb = size_bytes / 1024
		
		if size_kb <= 10:
			return 'small'
		elif size_kb <= 1000:
			return 'medium'
		else:
			return 'large'
	
	def _categorize_by_frequency(self, frequency: float) -> str:
		"""Categorize entry by access frequency"""
		if frequency >= 100:
			return 'high'
		elif frequency >= 10:
			return 'medium'
		else:
			return 'low'
	
	def _calculate_recency_score(self, last_accessed: Optional[datetime]) -> float:
		"""Calculate recency score based on last access time"""
		if not last_accessed:
			return 0.0
		
		hours_since_access = (datetime.utcnow() - last_accessed).total_seconds() / 3600
		
		# Exponential decay: recent accesses are much more valuable
		return math.exp(-hours_since_access / 24.0)  # Half-life of 24 hours
	
	def _generate_placement_reasoning(self, entry: CacheEntry, tier: CacheTier, 
									  analysis: Dict[str, Any]) -> str:
		"""Generate human-readable reasoning for placement decision"""
		
		reasons = []
		
		# Frequency-based reasoning
		if analysis['frequency_category'] == 'high':
			if tier in [CacheTier.L1, CacheTier.L2]:
				reasons.append("High access frequency requires fast tier")
		elif analysis['frequency_category'] == 'low':
			if tier in [CacheTier.L3, CacheTier.EDGE]:
				reasons.append("Low access frequency suitable for slower tier")
		
		# Size-based reasoning
		if analysis['size_category'] == 'large' and tier == CacheTier.L3:
			reasons.append("Large entry fits better in high-capacity tier")
		elif analysis['size_category'] == 'small' and tier == CacheTier.L1:
			reasons.append("Small entry can utilize fast memory tier")
		
		# Cost reasoning
		tier_config = self.tier_configs.get(tier)
		if tier_config and tier_config.cost_per_gb_per_hour < 0.05:
			reasons.append("Cost-effective tier for storage requirements")
		
		return "; ".join(reasons) if reasons else "General optimization alignment"
	
	def _generate_tier_reasoning(self, entry: CacheEntry, tier: CacheTier) -> str:
		"""Generate reasoning for specific tier recommendation"""
		
		tier_config = self.tier_configs.get(tier)
		if not tier_config:
			return "Configuration not available"
		
		return f"Tier {tier.value}: {tier_config.estimated_latency_ms}ms latency, " \
			   f"${tier_config.cost_per_gb_per_hour:.3f}/GB/hour"
	
	# Tier operation methods
	
	async def _write_to_tier(self, tier: CacheTier, key: str, entry: CacheEntry) -> bool:
		"""Write entry to specific tier"""
		# Placeholder - would write to actual backend
		return True
	
	async def _remove_from_tier(self, tier: CacheTier, key: str) -> bool:
		"""Remove entry from specific tier"""
		# Placeholder - would remove from actual backend
		return True
	
	async def _verify_tier_write(self, tier: CacheTier, key: str, entry: CacheEntry) -> bool:
		"""Verify write to tier was successful"""
		# Placeholder - would verify with actual backend
		return True
	
	async def _key_exists_in_tier(self, tier: CacheTier, key: str) -> bool:
		"""Check if key exists in specific tier"""
		# Placeholder - would check actual backend
		return self.data_placement.get(key) == tier
	
	# Consistency management methods
	
	async def _ensure_strong_consistency(self, key: str, tiers: List[CacheTier]) -> bool:
		"""Ensure strong consistency across tiers"""
		# Placeholder for strong consistency implementation
		return True
	
	async def _ensure_session_consistency(self, key: str, tiers: List[CacheTier]) -> bool:
		"""Ensure session consistency across tiers"""
		# Placeholder for session consistency implementation
		return True
	
	async def _ensure_monotonic_read_consistency(self, key: str, tiers: List[CacheTier]) -> bool:
		"""Ensure monotonic read consistency across tiers"""
		# Placeholder for monotonic read consistency implementation
		return True
	
	async def _ensure_causal_consistency(self, key: str, tiers: List[CacheTier]) -> bool:
		"""Ensure causal consistency across tiers"""
		# Placeholder for causal consistency implementation
		return True
	
	# Helper methods
	
	async def _validate_migration(self, key: str, source: CacheTier, target: CacheTier) -> bool:
		"""Validate that migration is safe and beneficial"""
		
		if source == target:
			return False
		
		# Check target tier capacity
		target_config = self.tier_configs.get(target)
		target_metrics = self.tier_metrics.get(target)
		
		if not target_config or not target_metrics:
			return False
		
		if target_metrics.utilization_percent > 90:
			return False
		
		return True
	
	async def _calculate_performance_gain(self, entry: CacheEntry, 
										  current_tier: Optional[CacheTier],
										  target_tier: CacheTier) -> float:
		"""Calculate expected performance gain from migration"""
		
		if not current_tier:
			return 10.0  # New placement is always beneficial
		
		current_config = self.tier_configs.get(current_tier)
		target_config = self.tier_configs.get(target_tier)
		
		if not current_config or not target_config:
			return 0.0
		
		# Calculate latency improvement
		latency_improvement = max(0, current_config.estimated_latency_ms - target_config.estimated_latency_ms)
		
		# Weight by access frequency
		weighted_improvement = latency_improvement * entry.access_frequency
		
		return min(weighted_improvement, 100.0)
	
	async def _calculate_migration_cost(self, entry: CacheEntry,
										current_tier: Optional[CacheTier], 
										target_tier: CacheTier) -> float:
		"""Calculate cost of migration"""
		
		# Base cost for network transfer
		base_cost = entry.size_bytes / (1024 * 1024)  # Cost per MB
		
		# Add tier-specific costs
		target_config = self.tier_configs.get(target_tier)
		if target_config:
			tier_cost = target_config.cost_per_gb_per_hour * (entry.size_bytes / (1024**3))
			base_cost += tier_cost
		
		return base_cost
	
	async def _update_tier_metrics_after_migration(self, source_tier: CacheTier,
												   target_tier: CacheTier, entry: CacheEntry) -> None:
		"""Update tier metrics after successful migration"""
		
		entry_size_mb = entry.size_bytes / (1024 * 1024)
		
		# Update source tier metrics
		source_metrics = self.tier_metrics.get(source_tier)
		if source_metrics:
			source_metrics.current_size_mb = max(0, source_metrics.current_size_mb - entry_size_mb)
			source_config = self.tier_configs.get(source_tier)
			if source_config:
				source_metrics.utilization_percent = (source_metrics.current_size_mb / source_config.capacity_mb) * 100
		
		# Update target tier metrics
		target_metrics = self.tier_metrics.get(target_tier)
		if target_metrics:
			target_metrics.current_size_mb += entry_size_mb
			target_config = self.tier_configs.get(target_tier)
			if target_config:
				target_metrics.utilization_percent = (target_metrics.current_size_mb / target_config.capacity_mb) * 100
	
	async def _rebalance_tier_loads(self) -> None:
		"""Rebalance loads across tiers"""
		
		# Update tier load balancing scores
		for tier in CacheTier:
			tier_metrics = self.tier_metrics.get(tier)
			if tier_metrics:
				self.tier_load_balancing[tier] = 1.0 - (tier_metrics.utilization_percent / 100.0)


# Export main components
__all__ = [
	'MultiTierCacheHierarchy',
	'TierStrategy',
	'ConsistencyLevel',
	'TierConfiguration',
	'TierMetrics',
	'DataPlacementDecision'
]