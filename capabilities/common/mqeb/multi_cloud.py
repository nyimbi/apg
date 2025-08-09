#!/usr/bin/env python3
"""
APG Message Queue Event Bus (MQEB) - Multi-Cloud Federation
Active-active multi-cloud messaging with intelligent failover

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import hashlib
import secrets
from uuid_extensions import uuid7str

from .models import MQMessage, TopicConfiguration
from .service import MQEBService


class CloudProvider(str, Enum):
	"""Supported cloud providers"""
	AWS = "aws"
	AZURE = "azure"
	GCP = "gcp"
	ALIBABA_CLOUD = "alibaba_cloud"
	IBM_CLOUD = "ibm_cloud"
	ORACLE_CLOUD = "oracle_cloud"
	PRIVATE_CLOUD = "private_cloud"


class ReplicationStrategy(str, Enum):
	"""Message replication strategies"""
	ACTIVE_ACTIVE = "active_active"      # All clouds active, full replication
	ACTIVE_PASSIVE = "active_passive"    # Primary + backup clouds
	GEOGRAPHIC = "geographic"            # Route by geography
	LATENCY_BASED = "latency_based"      # Route by lowest latency
	COST_OPTIMIZED = "cost_optimized"    # Route by cost efficiency
	COMPLIANCE_BASED = "compliance_based" # Route by data residency requirements


class FailoverTrigger(str, Enum):
	"""Triggers for automatic failover"""
	AVAILABILITY = "availability"        # Cloud region unavailable
	LATENCY = "latency"                 # High latency detected
	COST = "cost"                       # Cost threshold exceeded
	COMPLIANCE = "compliance"           # Compliance violation
	PERFORMANCE = "performance"         # Performance degradation
	MANUAL = "manual"                   # Manual failover trigger


@dataclass
class CloudRegion:
	"""Cloud region configuration"""
	region_id: str
	provider: CloudProvider
	region_name: str
	endpoint_url: str
	availability_zones: List[str]
	is_primary: bool = False
	enabled: bool = True
	latency_ms: float = 0.0
	cost_per_gb: float = 0.0
	compliance_zones: List[str] = field(default_factory=list)
	last_health_check: Optional[datetime] = None
	health_status: str = "unknown"  # healthy, degraded, unhealthy


@dataclass
class ReplicationRule:
	"""Message replication rule"""
	rule_id: str
	name: str
	topic_patterns: List[str]
	source_regions: List[str]
	target_regions: List[str]
	strategy: ReplicationStrategy
	consistency_level: str = "eventual"  # strong, eventual, weak
	max_replication_delay_ms: int = 5000
	priority: int = 100
	enabled: bool = True


@dataclass
class FailoverEvent:
	"""Failover event record"""
	event_id: str
	trigger: FailoverTrigger
	source_region: str
	target_region: str
	affected_topics: List[str]
	triggered_at: datetime
	completed_at: Optional[datetime] = None
	success: bool = False
	error_details: Optional[str] = None
	recovery_time_ms: Optional[float] = None


@dataclass
class CrossCloudMetrics:
	"""Cross-cloud performance metrics"""
	timestamp: datetime
	total_messages_replicated: int
	replication_lag_ms: float
	cross_cloud_bandwidth_mbps: float
	failover_events: int
	cost_per_hour: float
	compliance_violations: int
	availability_percentage: float


class CloudRegionManager:
	"""Manages cloud regions and their health"""
	
	def __init__(self):
		self.regions: Dict[str, CloudRegion] = {}
		self.health_check_interval = 30  # seconds
		self.latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
		
		self.logger = logging.getLogger('mqeb.cloud_region_manager')
	
	def add_region(self, region: CloudRegion) -> None:
		"""Add cloud region"""
		self.regions[region.region_id] = region
		self.logger.info(f"Added cloud region {region.region_id} ({region.provider.value})")
	
	def remove_region(self, region_id: str) -> bool:
		"""Remove cloud region"""
		if region_id in self.regions:
			del self.regions[region_id]
			self.logger.info(f"Removed cloud region {region_id}")
			return True
		return False
	
	async def health_check_region(self, region_id: str) -> bool:
		"""Perform health check on region"""
		try:
			region = self.regions.get(region_id)
			if not region:
				return False
			
			start_time = time.time()
			
			# Simulate health check (in production, would ping actual endpoints)
			await asyncio.sleep(0.01)  # Simulate network check
			
			# Measure latency
			latency_ms = (time.time() - start_time) * 1000
			region.latency_ms = latency_ms
			self.latency_history[region_id].append(latency_ms)
			
			# Update health status
			region.last_health_check = datetime.utcnow()
			
			if latency_ms < 50:
				region.health_status = "healthy"
			elif latency_ms < 200:
				region.health_status = "degraded"
			else:
				region.health_status = "unhealthy"
			
			return region.health_status in ["healthy", "degraded"]
			
		except Exception as e:
			self.logger.error(f"Health check failed for region {region_id}: {e}")
			if region_id in self.regions:
				self.regions[region_id].health_status = "unhealthy"
			return False
	
	async def health_check_all_regions(self) -> Dict[str, bool]:
		"""Health check all regions"""
		results = {}
		tasks = []
		
		for region_id in self.regions:
			task = asyncio.create_task(self.health_check_region(region_id))
			tasks.append((region_id, task))
		
		for region_id, task in tasks:
			try:
				results[region_id] = await task
			except Exception as e:
				self.logger.error(f"Health check failed for {region_id}: {e}")
				results[region_id] = False
		
		return results
	
	def get_healthy_regions(self) -> List[CloudRegion]:
		"""Get list of healthy regions"""
		return [
			region for region in self.regions.values()
			if region.enabled and region.health_status == "healthy"
		]
	
	def get_region_by_provider(self, provider: CloudProvider) -> List[CloudRegion]:
		"""Get regions by cloud provider"""
		return [
			region for region in self.regions.values()
			if region.provider == provider and region.enabled
		]
	
	def get_optimal_region(self, criteria: str = "latency") -> Optional[CloudRegion]:
		"""Get optimal region based on criteria"""
		healthy_regions = self.get_healthy_regions()
		if not healthy_regions:
			return None
		
		if criteria == "latency":
			return min(healthy_regions, key=lambda r: r.latency_ms)
		elif criteria == "cost":
			return min(healthy_regions, key=lambda r: r.cost_per_gb)
		elif criteria == "primary":
			primary_regions = [r for r in healthy_regions if r.is_primary]
			return primary_regions[0] if primary_regions else healthy_regions[0]
		
		return healthy_regions[0]


class MessageReplicator:
	"""Handles message replication across cloud regions"""
	
	def __init__(self, region_manager: CloudRegionManager):
		self.region_manager = region_manager
		self.replication_rules: Dict[str, ReplicationRule] = {}
		self.replication_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
		self.replication_stats = defaultdict(int)
		
		self.logger = logging.getLogger('mqeb.message_replicator')
	
	def add_replication_rule(self, rule: ReplicationRule) -> None:
		"""Add message replication rule"""
		self.replication_rules[rule.rule_id] = rule
		self.logger.info(f"Added replication rule {rule.rule_id}: {rule.name}")
	
	def remove_replication_rule(self, rule_id: str) -> bool:
		"""Remove replication rule"""
		if rule_id in self.replication_rules:
			del self.replication_rules[rule_id]
			self.logger.info(f"Removed replication rule {rule_id}")
			return True
		return False
	
	async def replicate_message(self, message: MQMessage, source_region: str) -> List[str]:
		"""Replicate message to target regions"""
		replicated_regions = []
		
		try:
			# Find applicable replication rules
			applicable_rules = self._find_applicable_rules(message, source_region)
			
			if not applicable_rules:
				return replicated_regions
			
			# Determine target regions for replication
			target_regions = set()
			for rule in applicable_rules:
				if rule.enabled:
					target_regions.update(rule.target_regions)
			
			# Remove source region from targets
			target_regions.discard(source_region)
			
			# Replicate to each target region
			replication_tasks = []
			for target_region in target_regions:
				task = asyncio.create_task(
					self._replicate_to_region(message, source_region, target_region)
				)
				replication_tasks.append((target_region, task))
			
			# Wait for all replications
			for target_region, task in replication_tasks:
				try:
					success = await task
					if success:
						replicated_regions.append(target_region)
						self.replication_stats[f'{source_region}_{target_region}'] += 1
				except Exception as e:
					self.logger.error(f"Replication to {target_region} failed: {e}")
			
			self.logger.debug(f"Message {message.id} replicated to {len(replicated_regions)} regions")
			
		except Exception as e:
			self.logger.error(f"Message replication failed: {e}")
		
		return replicated_regions
	
	def _find_applicable_rules(self, message: MQMessage, source_region: str) -> List[ReplicationRule]:
		"""Find replication rules applicable to message"""
		applicable_rules = []
		
		for rule in self.replication_rules.values():
			if not rule.enabled:
				continue
			
			# Check if source region matches
			if rule.source_regions and source_region not in rule.source_regions:
				continue
			
			# Check topic patterns
			import fnmatch
			topic_matches = any(
				fnmatch.fnmatch(message.topic, pattern)
				for pattern in rule.topic_patterns
			)
			
			if topic_matches:
				applicable_rules.append(rule)
		
		# Sort by priority (lower number = higher priority)
		applicable_rules.sort(key=lambda r: r.priority)
		return applicable_rules
	
	async def _replicate_to_region(self, message: MQMessage, source_region: str, target_region: str) -> bool:
		"""Replicate message to specific region"""
		try:
			# Get target region configuration
			region = self.region_manager.regions.get(target_region)
			if not region or not region.enabled:
				return False
			
			# Check region health
			if region.health_status != "healthy":
				self.logger.warning(f"Skipping replication to unhealthy region {target_region}")
				return False
			
			# Create replication payload
			replication_payload = {
				'message_id': message.id,
				'topic': message.topic,
				'payload': message.payload.decode('utf-8', errors='ignore'),
				'headers': message.headers,
				'timestamp': message.timestamp.isoformat(),
				'source_region': source_region,
				'target_region': target_region,
				'replication_timestamp': datetime.utcnow().isoformat()
			}
			
			# Queue for replication (in production, would send to actual cloud service)
			self.replication_queues[target_region].append(replication_payload)
			
			# Simulate network latency
			await asyncio.sleep(0.01)
			
			self.logger.debug(f"Message {message.id} replicated: {source_region} -> {target_region}")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to replicate to {target_region}: {e}")
			return False
	
	async def get_replication_stats(self) -> Dict[str, Any]:
		"""Get replication statistics"""
		total_replications = sum(self.replication_stats.values())
		
		return {
			'total_replications': total_replications,
			'replication_routes': dict(self.replication_stats),
			'active_rules': len([r for r in self.replication_rules.values() if r.enabled]),
			'queue_sizes': {region: len(queue) for region, queue in self.replication_queues.items()}
		}


class FailoverManager:
	"""Manages automatic failover between cloud regions"""
	
	def __init__(self, region_manager: CloudRegionManager):
		self.region_manager = region_manager
		self.failover_events: List[FailoverEvent] = []
		self.failover_thresholds = {
			FailoverTrigger.AVAILABILITY: 0.99,  # 99% availability threshold
			FailoverTrigger.LATENCY: 500.0,      # 500ms latency threshold
			FailoverTrigger.PERFORMANCE: 0.80    # 80% performance threshold
		}
		self.cooldown_period_seconds = 300  # 5 minutes between failovers
		
		self.logger = logging.getLogger('mqeb.failover_manager')
	
	async def evaluate_failover_conditions(self) -> List[FailoverEvent]:
		"""Evaluate conditions that might trigger failover"""
		potential_failovers = []
		
		try:
			# Check each region for failover conditions
			for region_id, region in self.region_manager.regions.items():
				if not region.is_primary or not region.enabled:
					continue
				
				# Check availability
				if region.health_status == "unhealthy":
					failover_event = await self._create_failover_event(
						FailoverTrigger.AVAILABILITY,
						region_id,
						f"Region {region_id} is unhealthy"
					)
					if failover_event:
						potential_failovers.append(failover_event)
				
				# Check latency
				elif region.latency_ms > self.failover_thresholds[FailoverTrigger.LATENCY]:
					failover_event = await self._create_failover_event(
						FailoverTrigger.LATENCY,
						region_id,
						f"High latency: {region.latency_ms}ms"
					)
					if failover_event:
						potential_failovers.append(failover_event)
		
		except Exception as e:
			self.logger.error(f"Failover evaluation failed: {e}")
		
		return potential_failovers
	
	async def _create_failover_event(self, trigger: FailoverTrigger, source_region: str, reason: str) -> Optional[FailoverEvent]:
		"""Create failover event"""
		try:
			# Check cooldown period
			recent_failovers = [
				event for event in self.failover_events
				if (event.source_region == source_region and
					(datetime.utcnow() - event.triggered_at).total_seconds() < self.cooldown_period_seconds)
			]
			
			if recent_failovers:
				self.logger.debug(f"Failover for {source_region} in cooldown period")
				return None
			
			# Find target region
			target_region = self._select_failover_target(source_region)
			if not target_region:
				self.logger.error(f"No suitable failover target for {source_region}")
				return None
			
			# Create failover event
			failover_event = FailoverEvent(
				event_id=f"failover_{uuid7str()}",
				trigger=trigger,
				source_region=source_region,
				target_region=target_region,
				affected_topics=["*"],  # All topics affected
				triggered_at=datetime.utcnow()
			)
			
			self.logger.warning(f"Failover event created: {source_region} -> {target_region} ({reason})")
			return failover_event
			
		except Exception as e:
			self.logger.error(f"Failed to create failover event: {e}")
			return None
	
	def _select_failover_target(self, source_region: str) -> Optional[str]:
		"""Select best failover target region"""
		# Get healthy regions excluding source
		healthy_regions = [
			region for region in self.region_manager.get_healthy_regions()
			if region.region_id != source_region
		]
		
		if not healthy_regions:
			return None
		
		# Prefer regions from different cloud providers for diversity
		source_provider = self.region_manager.regions[source_region].provider
		different_provider_regions = [
			region for region in healthy_regions
			if region.provider != source_provider
		]
		
		if different_provider_regions:
			# Select region with lowest latency from different provider
			return min(different_provider_regions, key=lambda r: r.latency_ms).region_id
		else:
			# Select region with lowest latency from same provider
			return min(healthy_regions, key=lambda r: r.latency_ms).region_id
	
	async def execute_failover(self, failover_event: FailoverEvent) -> bool:
		"""Execute failover operation"""
		start_time = time.time()
		
		try:
			self.logger.warning(f"Executing failover {failover_event.event_id}: {failover_event.source_region} -> {failover_event.target_region}")
			
			# Disable source region
			source_region = self.region_manager.regions.get(failover_event.source_region)
			if source_region:
				source_region.enabled = False
				source_region.is_primary = False
			
			# Promote target region to primary
			target_region = self.region_manager.regions.get(failover_event.target_region)
			if target_region:
				target_region.is_primary = True
				target_region.enabled = True
			
			# Simulate failover operations (DNS updates, load balancer changes, etc.)
			await asyncio.sleep(0.1)
			
			# Mark failover as completed
			failover_event.completed_at = datetime.utcnow()
			failover_event.success = True
			failover_event.recovery_time_ms = (time.time() - start_time) * 1000
			
			# Record failover event
			self.failover_events.append(failover_event)
			
			self.logger.info(f"Failover {failover_event.event_id} completed in {failover_event.recovery_time_ms:.2f}ms")
			return True
			
		except Exception as e:
			failover_event.error_details = str(e)
			failover_event.success = False
			self.failover_events.append(failover_event)
			
			self.logger.error(f"Failover {failover_event.event_id} failed: {e}")
			return False
	
	async def manual_failover(self, source_region: str, target_region: str, reason: str) -> bool:
		"""Trigger manual failover"""
		failover_event = FailoverEvent(
			event_id=f"manual_{uuid7str()}",
			trigger=FailoverTrigger.MANUAL,
			source_region=source_region,
			target_region=target_region,
			affected_topics=["*"],
			triggered_at=datetime.utcnow()
		)
		
		return await self.execute_failover(failover_event)
	
	def get_failover_history(self, hours: int = 24) -> List[FailoverEvent]:
		"""Get failover history"""
		cutoff_time = datetime.utcnow() - timedelta(hours=hours)
		return [
			event for event in self.failover_events
			if event.triggered_at > cutoff_time
		]


class MultiCloudFederation:
	"""Main multi-cloud federation engine"""
	
	def __init__(self, mqeb_service: MQEBService):
		self.service = mqeb_service
		self.region_manager = CloudRegionManager()
		self.message_replicator = MessageReplicator(self.region_manager)
		self.failover_manager = FailoverManager(self.region_manager)
		
		# Configuration
		self.enabled = True
		self.current_region = "primary"
		self.federation_strategy = ReplicationStrategy.ACTIVE_ACTIVE
		
		# Metrics
		self.federation_metrics: List[CrossCloudMetrics] = []
		self.performance_stats = defaultdict(float)
		
		# Background tasks
		self._background_tasks: Set[asyncio.Task] = set()
		
		self.logger = logging.getLogger('mqeb.multi_cloud_federation')
	
	async def initialize(self) -> None:
		"""Initialize multi-cloud federation"""
		self.logger.info("Initializing multi-cloud federation...")
		
		# Initialize default regions
		await self._initialize_default_regions()
		
		# Start background tasks
		await self._start_background_tasks()
		
		self.logger.info("Multi-cloud federation initialized")
	
	async def shutdown(self) -> None:
		"""Shutdown multi-cloud federation"""
		self.enabled = False
		
		# Cancel background tasks
		for task in self._background_tasks:
			task.cancel()
		
		await asyncio.gather(*self._background_tasks, return_exceptions=True)
		self.logger.info("Multi-cloud federation shut down")
	
	async def _initialize_default_regions(self) -> None:
		"""Initialize default cloud regions"""
		default_regions = [
			CloudRegion(
				region_id="aws_us_east_1",
				provider=CloudProvider.AWS,
				region_name="US East (N. Virginia)",
				endpoint_url="https://mqeb.us-east-1.amazonaws.com",
				availability_zones=["us-east-1a", "us-east-1b", "us-east-1c"],
				is_primary=True,
				cost_per_gb=0.023,
				compliance_zones=["usa", "north_america"]
			),
			CloudRegion(
				region_id="gcp_us_central_1",
				provider=CloudProvider.GCP,
				region_name="US Central 1",
				endpoint_url="https://mqeb.us-central1.gcp.com",
				availability_zones=["us-central1-a", "us-central1-b", "us-central1-c"],
				cost_per_gb=0.020,
				compliance_zones=["usa", "north_america"]
			),
			CloudRegion(
				region_id="azure_eastus",
				provider=CloudProvider.AZURE,
				region_name="East US",
				endpoint_url="https://mqeb.eastus.azure.com",
				availability_zones=["eastus-1", "eastus-2", "eastus-3"],
				cost_per_gb=0.024,
				compliance_zones=["usa", "north_america"]
			),
			CloudRegion(
				region_id="aws_eu_west_1",
				provider=CloudProvider.AWS,
				region_name="EU West (Ireland)",
				endpoint_url="https://mqeb.eu-west-1.amazonaws.com",
				availability_zones=["eu-west-1a", "eu-west-1b", "eu-west-1c"],
				cost_per_gb=0.025,
				compliance_zones=["eu", "gdpr"]
			)
		]
		
		for region in default_regions:
			self.region_manager.add_region(region)
		
		# Create default replication rules
		await self._create_default_replication_rules()
	
	async def _create_default_replication_rules(self) -> None:
		"""Create default replication rules"""
		default_rules = [
			ReplicationRule(
				rule_id="global_replication",
				name="Global Active-Active Replication",
				topic_patterns=["*"],
				source_regions=["aws_us_east_1"],
				target_regions=["gcp_us_central_1", "azure_eastus"],
				strategy=ReplicationStrategy.ACTIVE_ACTIVE,
				priority=100
			),
			ReplicationRule(
				rule_id="eu_compliance_replication",
				name="EU Data Residency Replication",
				topic_patterns=["eu.*", "gdpr.*", "customer.eu.*"],
				source_regions=["aws_eu_west_1"],
				target_regions=["aws_eu_west_1"],
				strategy=ReplicationStrategy.COMPLIANCE_BASED,
				priority=10
			),
			ReplicationRule(
				rule_id="critical_data_replication",
				name="Critical Data Multi-Region",
				topic_patterns=["financial.*", "payment.*", "critical.*"],
				source_regions=["aws_us_east_1"],
				target_regions=["gcp_us_central_1", "azure_eastus", "aws_eu_west_1"],
				strategy=ReplicationStrategy.ACTIVE_ACTIVE,
				max_replication_delay_ms=1000,
				priority=1
			)
		]
		
		for rule in default_rules:
			self.message_replicator.add_replication_rule(rule)
	
	async def federate_message(self, message: MQMessage) -> Dict[str, Any]:
		"""Federate message across cloud regions"""
		federation_result = {
			'message_id': message.id,
			'source_region': self.current_region,
			'replicated_regions': [],
			'replication_time_ms': 0,
			'success': False
		}
		
		try:
			if not self.enabled:
				federation_result['success'] = True
				return federation_result
			
			start_time = time.time()
			
			# Replicate message to target regions
			replicated_regions = await self.message_replicator.replicate_message(
				message, self.current_region
			)
			
			federation_result['replicated_regions'] = replicated_regions
			federation_result['replication_time_ms'] = (time.time() - start_time) * 1000
			federation_result['success'] = True
			
			# Update performance stats
			self.performance_stats['total_replications'] += len(replicated_regions)
			self.performance_stats['avg_replication_time_ms'] = (
				(self.performance_stats['avg_replication_time_ms'] * 0.9) + 
				(federation_result['replication_time_ms'] * 0.1)
			)
			
			self.logger.debug(f"Message {message.id} federated to {len(replicated_regions)} regions")
			
		except Exception as e:
			federation_result['error'] = str(e)
			self.logger.error(f"Message federation failed: {e}")
		
		return federation_result
	
	async def _start_background_tasks(self) -> None:
		"""Start background tasks"""
		
		# Region health monitoring
		task = asyncio.create_task(self._region_health_monitoring_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Failover monitoring
		task = asyncio.create_task(self._failover_monitoring_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Metrics collection
		task = asyncio.create_task(self._metrics_collection_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
	
	async def _region_health_monitoring_loop(self) -> None:
		"""Background task for region health monitoring"""
		while self.enabled:
			try:
				await asyncio.sleep(30)  # Check every 30 seconds
				
				# Health check all regions
				health_results = await self.region_manager.health_check_all_regions()
				
				unhealthy_regions = [
					region_id for region_id, is_healthy in health_results.items()
					if not is_healthy
				]
				
				if unhealthy_regions:
					self.logger.warning(f"Unhealthy regions detected: {unhealthy_regions}")
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Region health monitoring error: {e}")
	
	async def _failover_monitoring_loop(self) -> None:
		"""Background task for failover monitoring"""
		while self.enabled:
			try:
				await asyncio.sleep(60)  # Check every minute
				
				# Evaluate failover conditions
				potential_failovers = await self.failover_manager.evaluate_failover_conditions()
				
				# Execute automatic failovers
				for failover_event in potential_failovers:
					if failover_event.trigger != FailoverTrigger.MANUAL:
						success = await self.failover_manager.execute_failover(failover_event)
						if success:
							self.current_region = failover_event.target_region
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Failover monitoring error: {e}")
	
	async def _metrics_collection_loop(self) -> None:
		"""Background task for metrics collection"""
		while self.enabled:
			try:
				await asyncio.sleep(300)  # Collect every 5 minutes
				
				# Collect cross-cloud metrics
				replication_stats = await self.message_replicator.get_replication_stats()
				
				metrics = CrossCloudMetrics(
					timestamp=datetime.utcnow(),
					total_messages_replicated=replication_stats['total_replications'],
					replication_lag_ms=self.performance_stats.get('avg_replication_time_ms', 0),
					cross_cloud_bandwidth_mbps=100.0,  # Simplified
					failover_events=len(self.failover_manager.get_failover_history(hours=1)),
					cost_per_hour=self._calculate_hourly_cost(),
					compliance_violations=0,  # Would be calculated from compliance engine
					availability_percentage=self._calculate_availability()
				)
				
				self.federation_metrics.append(metrics)
				
				# Keep only last 24 hours of metrics
				cutoff_time = datetime.utcnow() - timedelta(hours=24)
				self.federation_metrics = [
					m for m in self.federation_metrics
					if m.timestamp > cutoff_time
				]
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Metrics collection error: {e}")
	
	def _calculate_hourly_cost(self) -> float:
		"""Calculate hourly cost across all regions"""
		total_cost = 0.0
		for region in self.region_manager.regions.values():
			if region.enabled:
				# Simplified cost calculation
				total_cost += region.cost_per_gb * 0.1  # Assume 0.1 GB/hour per region
		return total_cost
	
	def _calculate_availability(self) -> float:
		"""Calculate overall system availability"""
		healthy_regions = len(self.region_manager.get_healthy_regions())
		total_regions = len([r for r in self.region_manager.regions.values() if r.enabled])
		
		if total_regions == 0:
			return 0.0
		
		return (healthy_regions / total_regions) * 100.0
	
	async def get_federation_status(self) -> Dict[str, Any]:
		"""Get federation status"""
		replication_stats = await self.message_replicator.get_replication_stats()
		recent_failovers = self.failover_manager.get_failover_history(hours=24)
		
		return {
			'enabled': self.enabled,
			'current_region': self.current_region,
			'federation_strategy': self.federation_strategy.value,
			'total_regions': len(self.region_manager.regions),
			'healthy_regions': len(self.region_manager.get_healthy_regions()),
			'replication_stats': replication_stats,
			'recent_failovers': len(recent_failovers),
			'performance_stats': dict(self.performance_stats),
			'regions': [
				{
					'region_id': region.region_id,
					'provider': region.provider.value,
					'region_name': region.region_name,
					'health_status': region.health_status,
					'latency_ms': region.latency_ms,
					'is_primary': region.is_primary,
					'enabled': region.enabled
				}
				for region in self.region_manager.regions.values()
			]
		}


# Factory function
async def create_multi_cloud_federation(mqeb_service: MQEBService) -> MultiCloudFederation:
	"""Create and initialize multi-cloud federation"""
	federation = MultiCloudFederation(mqeb_service)
	await federation.initialize()
	return federation


# Export components
__all__ = [
	'MultiCloudFederation', 'CloudRegionManager', 'MessageReplicator', 'FailoverManager',
	'CloudProvider', 'ReplicationStrategy', 'FailoverTrigger',
	'CloudRegion', 'ReplicationRule', 'FailoverEvent', 'CrossCloudMetrics',
	'create_multi_cloud_federation'
]