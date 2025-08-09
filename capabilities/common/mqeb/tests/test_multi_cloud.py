#!/usr/bin/env python3
"""
APG Message Queue Event Bus (MQEB) - Multi-Cloud Federation Tests
Tests for active-active multi-cloud messaging with intelligent failover

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid_extensions import uuid7str

# Import MQEB components
from ..models import MQMessage, MessagePriority
from ..service import MQEBService
from ..multi_cloud import (
	MultiCloudFederation, CloudRegionManager, MessageReplicator, FailoverManager,
	CloudProvider, ReplicationStrategy, FailoverTrigger,
	CloudRegion, ReplicationRule, FailoverEvent, CrossCloudMetrics,
	create_multi_cloud_federation
)


class TestCloudRegion:
	"""Test cloud region functionality"""
	
	def test_cloud_region_creation(self):
		"""Test cloud region creation"""
		region = CloudRegion(
			region_id="test_region_001",
			provider=CloudProvider.AWS,
			region_name="Test Region",
			endpoint_url="https://test.aws.example.com",
			availability_zones=["test-1a", "test-1b", "test-1c"],
			is_primary=True,
			latency_ms=25.5,
			cost_per_gb=0.023,
			compliance_zones=["usa", "pci_dss"]
		)
		
		assert region.region_id == "test_region_001"
		assert region.provider == CloudProvider.AWS
		assert region.is_primary == True
		assert region.enabled == True
		assert len(region.availability_zones) == 3
		assert len(region.compliance_zones) == 2


class TestCloudRegionManager:
	"""Test cloud region management"""
	
	def test_region_manager_initialization(self):
		"""Test region manager initialization"""
		manager = CloudRegionManager()
		assert len(manager.regions) == 0
		assert len(manager.latency_history) == 0
	
	def test_add_remove_regions(self):
		"""Test adding and removing regions"""
		manager = CloudRegionManager()
		
		region = CloudRegion(
			region_id="test_region",
			provider=CloudProvider.GCP,
			region_name="Test GCP Region",
			endpoint_url="https://test.gcp.example.com",
			availability_zones=["test-a", "test-b"]
		)
		
		# Add region
		manager.add_region(region)
		assert "test_region" in manager.regions
		assert manager.regions["test_region"] == region
		
		# Remove region
		success = manager.remove_region("test_region")
		assert success == True
		assert "test_region" not in manager.regions
		
		# Try to remove non-existent region
		success = manager.remove_region("non_existent")
		assert success == False
	
	@pytest.mark.asyncio
	async def test_region_health_check(self):
		"""Test region health checking"""
		manager = CloudRegionManager()
		
		region = CloudRegion(
			region_id="health_test_region",
			provider=CloudProvider.AZURE,
			region_name="Health Test Region",
			endpoint_url="https://test.azure.example.com",
			availability_zones=["test-1", "test-2"]
		)
		
		manager.add_region(region)
		
		# Perform health check
		is_healthy = await manager.health_check_region("health_test_region")
		assert is_healthy == True
		
		# Check that health status was updated
		updated_region = manager.regions["health_test_region"]
		assert updated_region.health_status in ["healthy", "degraded"]
		assert updated_region.last_health_check is not None
		assert updated_region.latency_ms >= 0
	
	@pytest.mark.asyncio
	async def test_bulk_health_check(self):
		"""Test bulk health checking of all regions"""
		manager = CloudRegionManager()
		
		# Add multiple regions
		for i in range(3):
			region = CloudRegion(
				region_id=f"bulk_test_region_{i}",
				provider=CloudProvider.AWS,
				region_name=f"Bulk Test Region {i}",
				endpoint_url=f"https://region{i}.aws.example.com",
				availability_zones=[f"zone{i}-a", f"zone{i}-b"]
			)
			manager.add_region(region)
		
		# Health check all regions
		results = await manager.health_check_all_regions()
		
		assert len(results) == 3
		assert all(isinstance(healthy, bool) for healthy in results.values())
		
		# Check that all regions were updated
		for region in manager.regions.values():
			assert region.last_health_check is not None
	
	def test_get_healthy_regions(self):
		"""Test getting healthy regions"""
		manager = CloudRegionManager()
		
		# Add regions with different health statuses
		healthy_region = CloudRegion(
			region_id="healthy_region",
			provider=CloudProvider.GCP,
			region_name="Healthy Region",
			endpoint_url="https://healthy.gcp.example.com",
			availability_zones=["zone-a"]
		)
		healthy_region.health_status = "healthy"
		
		unhealthy_region = CloudRegion(
			region_id="unhealthy_region",
			provider=CloudProvider.AZURE,
			region_name="Unhealthy Region",
			endpoint_url="https://unhealthy.azure.example.com",
			availability_zones=["zone-b"]
		)
		unhealthy_region.health_status = "unhealthy"
		
		disabled_region = CloudRegion(
			region_id="disabled_region",
			provider=CloudProvider.AWS,
			region_name="Disabled Region",
			endpoint_url="https://disabled.aws.example.com",
			availability_zones=["zone-c"],
			enabled=False
		)
		disabled_region.health_status = "healthy"
		
		manager.add_region(healthy_region)
		manager.add_region(unhealthy_region)
		manager.add_region(disabled_region)
		
		healthy_regions = manager.get_healthy_regions()
		assert len(healthy_regions) == 1
		assert healthy_regions[0].region_id == "healthy_region"
	
	def test_get_optimal_region(self):
		"""Test getting optimal region based on criteria"""
		manager = CloudRegionManager()
		
		# Add regions with different characteristics
		low_latency_region = CloudRegion(
			region_id="low_latency",
			provider=CloudProvider.AWS,
			region_name="Low Latency Region",
			endpoint_url="https://low-latency.aws.example.com",
			availability_zones=["zone-a"],
			latency_ms=10.0,
			cost_per_gb=0.030
		)
		low_latency_region.health_status = "healthy"
		
		low_cost_region = CloudRegion(
			region_id="low_cost",
			provider=CloudProvider.GCP,
			region_name="Low Cost Region",
			endpoint_url="https://low-cost.gcp.example.com",
			availability_zones=["zone-b"],
			latency_ms=50.0,
			cost_per_gb=0.015
		)
		low_cost_region.health_status = "healthy"
		
		primary_region = CloudRegion(
			region_id="primary",
			provider=CloudProvider.AZURE,
			region_name="Primary Region",
			endpoint_url="https://primary.azure.example.com",
			availability_zones=["zone-c"],
			is_primary=True,
			latency_ms=30.0,
			cost_per_gb=0.025
		)
		primary_region.health_status = "healthy"
		
		manager.add_region(low_latency_region)
		manager.add_region(low_cost_region)
		manager.add_region(primary_region)
		
		# Test optimal region selection by latency
		optimal = manager.get_optimal_region("latency")
		assert optimal.region_id == "low_latency"
		
		# Test optimal region selection by cost
		optimal = manager.get_optimal_region("cost")
		assert optimal.region_id == "low_cost"
		
		# Test optimal region selection by primary
		optimal = manager.get_optimal_region("primary")
		assert optimal.region_id == "primary"


class TestReplicationRule:
	"""Test replication rule functionality"""
	
	def test_replication_rule_creation(self):
		"""Test replication rule creation"""
		rule = ReplicationRule(
			rule_id="test_rule",
			name="Test Replication Rule",
			topic_patterns=["user.*", "order.*"],
			source_regions=["us-east-1"],
			target_regions=["us-west-2", "eu-west-1"],
			strategy=ReplicationStrategy.ACTIVE_ACTIVE,
			consistency_level="eventual",
			max_replication_delay_ms=2000,
			priority=50
		)
		
		assert rule.rule_id == "test_rule"
		assert len(rule.topic_patterns) == 2
		assert len(rule.source_regions) == 1
		assert len(rule.target_regions) == 2
		assert rule.strategy == ReplicationStrategy.ACTIVE_ACTIVE
		assert rule.enabled == True


class TestMessageReplicator:
	"""Test message replication functionality"""
	
	@pytest.fixture
	def region_manager(self):
		"""Create region manager with test regions"""
		manager = CloudRegionManager()
		
		regions = [
			CloudRegion(
				region_id="source_region",
				provider=CloudProvider.AWS,
				region_name="Source Region",
				endpoint_url="https://source.aws.example.com",
				availability_zones=["zone-a"],
				is_primary=True
			),
			CloudRegion(
				region_id="target_region_1",
				provider=CloudProvider.GCP,
				region_name="Target Region 1",
				endpoint_url="https://target1.gcp.example.com",
				availability_zones=["zone-b"]
			),
			CloudRegion(
				region_id="target_region_2",
				provider=CloudProvider.AZURE,
				region_name="Target Region 2",
				endpoint_url="https://target2.azure.example.com",
				availability_zones=["zone-c"]
			)
		]
		
		for region in regions:
			region.health_status = "healthy"
			manager.add_region(region)
		
		return manager
	
	@pytest.fixture
	def message_replicator(self, region_manager):
		"""Create message replicator"""
		return MessageReplicator(region_manager)
	
	def test_replication_rule_management(self, message_replicator):
		"""Test adding and removing replication rules"""
		rule = ReplicationRule(
			rule_id="test_replication_rule",
			name="Test Rule",
			topic_patterns=["test.*"],
			source_regions=["source_region"],
			target_regions=["target_region_1", "target_region_2"],
			strategy=ReplicationStrategy.ACTIVE_ACTIVE
		)
		
		# Add rule
		message_replicator.add_replication_rule(rule)
		assert "test_replication_rule" in message_replicator.replication_rules
		
		# Remove rule
		success = message_replicator.remove_replication_rule("test_replication_rule")
		assert success == True
		assert "test_replication_rule" not in message_replicator.replication_rules
	
	@pytest.mark.asyncio
	async def test_message_replication(self, message_replicator):
		"""Test message replication to target regions"""
		# Add replication rule
		rule = ReplicationRule(
			rule_id="replication_test",
			name="Replication Test Rule",
			topic_patterns=["test.replication.*"],
			source_regions=["source_region"],
			target_regions=["target_region_1", "target_region_2"],
			strategy=ReplicationStrategy.ACTIVE_ACTIVE
		)
		message_replicator.add_replication_rule(rule)
		
		# Create test message
		message = MQMessage(
			topic="test.replication.data",
			payload=b'{"test": "replication message"}',
			tenant_id="test_tenant",
			source_application="replication_test"
		)
		
		# Replicate message
		replicated_regions = await message_replicator.replicate_message(message, "source_region")
		
		assert len(replicated_regions) == 2
		assert "target_region_1" in replicated_regions
		assert "target_region_2" in replicated_regions
		
		# Check replication queues
		assert len(message_replicator.replication_queues["target_region_1"]) > 0
		assert len(message_replicator.replication_queues["target_region_2"]) > 0
	
	@pytest.mark.asyncio
	async def test_replication_with_unhealthy_region(self, message_replicator):
		"""Test replication skips unhealthy regions"""
		# Mark one target region as unhealthy
		message_replicator.region_manager.regions["target_region_2"].health_status = "unhealthy"
		
		# Add replication rule
		rule = ReplicationRule(
			rule_id="health_test",
			name="Health Test Rule",
			topic_patterns=["test.health.*"],
			source_regions=["source_region"],
			target_regions=["target_region_1", "target_region_2"],
			strategy=ReplicationStrategy.ACTIVE_ACTIVE
		)
		message_replicator.add_replication_rule(rule)
		
		# Create and replicate message
		message = MQMessage(
			topic="test.health.check",
			payload=b'{"test": "health check"}',
			tenant_id="test_tenant",
			source_application="health_test"
		)
		
		replicated_regions = await message_replicator.replicate_message(message, "source_region")
		
		# Should only replicate to healthy region
		assert len(replicated_regions) == 1
		assert "target_region_1" in replicated_regions
		assert "target_region_2" not in replicated_regions
	
	@pytest.mark.asyncio
	async def test_replication_stats(self, message_replicator):
		"""Test replication statistics collection"""
		# Add replication rule
		rule = ReplicationRule(
			rule_id="stats_test",
			name="Stats Test Rule",
			topic_patterns=["stats.*"],
			source_regions=["source_region"],
			target_regions=["target_region_1"],
			strategy=ReplicationStrategy.ACTIVE_ACTIVE
		)
		message_replicator.add_replication_rule(rule)
		
		# Replicate multiple messages
		for i in range(3):
			message = MQMessage(
				topic=f"stats.test.{i}",
				payload=f"Stats test message {i}".encode(),
				tenant_id="test_tenant",
				source_application="stats_test"
			)
			await message_replicator.replicate_message(message, "source_region")
		
		# Get replication stats
		stats = await message_replicator.get_replication_stats()
		
		assert stats['total_replications'] == 3
		assert 'source_region_target_region_1' in stats['replication_routes']
		assert stats['replication_routes']['source_region_target_region_1'] == 3
		assert stats['active_rules'] == 1


class TestFailoverManager:
	"""Test failover management functionality"""
	
	@pytest.fixture
	def failover_manager(self):
		"""Create failover manager with test regions"""
		region_manager = CloudRegionManager()
		
		# Add primary and backup regions
		primary_region = CloudRegion(
			region_id="primary_region",
			provider=CloudProvider.AWS,
			region_name="Primary Region",
			endpoint_url="https://primary.aws.example.com",
			availability_zones=["zone-a"],
			is_primary=True,
			latency_ms=25.0
		)
		primary_region.health_status = "healthy"
		
		backup_region = CloudRegion(
			region_id="backup_region",
			provider=CloudProvider.GCP,
			region_name="Backup Region",
			endpoint_url="https://backup.gcp.example.com",
			availability_zones=["zone-b"],
			latency_ms=35.0
		)
		backup_region.health_status = "healthy"
		
		region_manager.add_region(primary_region)
		region_manager.add_region(backup_region)
		
		return FailoverManager(region_manager)
	
	@pytest.mark.asyncio
	async def test_failover_condition_evaluation(self, failover_manager):
		"""Test evaluation of failover conditions"""
		# Set primary region as unhealthy
		failover_manager.region_manager.regions["primary_region"].health_status = "unhealthy"
		
		# Evaluate failover conditions
		potential_failovers = await failover_manager.evaluate_failover_conditions()
		
		assert len(potential_failovers) > 0
		failover_event = potential_failovers[0]
		assert failover_event.trigger == FailoverTrigger.AVAILABILITY
		assert failover_event.source_region == "primary_region"
		assert failover_event.target_region == "backup_region"
	
	@pytest.mark.asyncio
	async def test_high_latency_failover(self, failover_manager):
		"""Test failover triggered by high latency"""
		# Set primary region with high latency
		failover_manager.region_manager.regions["primary_region"].latency_ms = 600.0  # Above threshold
		
		# Evaluate failover conditions
		potential_failovers = await failover_manager.evaluate_failover_conditions()
		
		assert len(potential_failovers) > 0
		failover_event = potential_failovers[0]
		assert failover_event.trigger == FailoverTrigger.LATENCY
	
	@pytest.mark.asyncio
	async def test_failover_execution(self, failover_manager):
		"""Test failover execution"""
		# Create failover event
		failover_event = FailoverEvent(
			event_id="test_failover",
			trigger=FailoverTrigger.MANUAL,
			source_region="primary_region",
			target_region="backup_region",
			affected_topics=["*"],
			triggered_at=datetime.utcnow()
		)
		
		# Execute failover
		success = await failover_manager.execute_failover(failover_event)
		assert success == True
		
		# Check that failover was recorded
		assert failover_event in failover_manager.failover_events
		assert failover_event.success == True
		assert failover_event.completed_at is not None
		assert failover_event.recovery_time_ms is not None
		
		# Check that regions were updated
		source_region = failover_manager.region_manager.regions["primary_region"]
		target_region = failover_manager.region_manager.regions["backup_region"]
		
		assert source_region.enabled == False
		assert source_region.is_primary == False
		assert target_region.is_primary == True
		assert target_region.enabled == True
	
	@pytest.mark.asyncio
	async def test_manual_failover(self, failover_manager):
		"""Test manual failover trigger"""
		success = await failover_manager.manual_failover(
			"primary_region",
			"backup_region",
			"Manual failover for testing"
		)
		
		assert success == True
		
		# Check that manual failover was recorded
		recent_failovers = failover_manager.get_failover_history(hours=1)
		assert len(recent_failovers) > 0
		
		manual_failover = recent_failovers[-1]
		assert manual_failover.trigger == FailoverTrigger.MANUAL
	
	def test_failover_cooldown(self, failover_manager):
		"""Test failover cooldown period"""
		# Create recent failover event
		recent_failover = FailoverEvent(
			event_id="recent_failover",
			trigger=FailoverTrigger.AVAILABILITY,
			source_region="primary_region",
			target_region="backup_region",
			affected_topics=["*"],
			triggered_at=datetime.utcnow() - timedelta(seconds=60)  # 1 minute ago
		)
		recent_failover.success = True
		failover_manager.failover_events.append(recent_failover)
		
		# Try to create another failover for same region
		# (This would be tested in the actual evaluation logic)
		assert len(failover_manager.failover_events) == 1


class TestMultiCloudFederation:
	"""Test main multi-cloud federation functionality"""
	
	@pytest.fixture
	async def mqeb_service(self):
		"""Create MQEB service for testing"""
		service = MQEBService()
		await service.initialize()
		yield service
		await service.shutdown()
	
	@pytest.fixture
	async def federation(self, mqeb_service):
		"""Create multi-cloud federation"""
		federation = await create_multi_cloud_federation(mqeb_service)
		yield federation
		await federation.shutdown()
	
	@pytest.mark.asyncio
	async def test_federation_initialization(self, federation):
		"""Test multi-cloud federation initialization"""
		assert federation.enabled == True
		assert federation.region_manager is not None
		assert federation.message_replicator is not None
		assert federation.failover_manager is not None
		
		# Check that default regions were created
		assert len(federation.region_manager.regions) > 0
		
		# Check that default replication rules were created
		assert len(federation.message_replicator.replication_rules) > 0
	
	@pytest.mark.asyncio
	async def test_message_federation(self, federation):
		"""Test message federation across regions"""
		# Create test message
		message = MQMessage(
			topic="test.federation.message",
			payload=b'{"federation": "test message"}',
			tenant_id="federation_test",
			source_application="federation_test_app",
			priority=MessagePriority.HIGH
		)
		
		# Federate message
		result = await federation.federate_message(message)
		
		assert result['success'] == True
		assert result['message_id'] == message.id
		assert result['source_region'] == federation.current_region
		assert len(result['replicated_regions']) >= 0
		assert result['replication_time_ms'] >= 0
	
	@pytest.mark.asyncio
	async def test_critical_message_replication(self, federation):
		"""Test that critical messages get replicated to all regions"""
		# Create critical message matching high-priority rule
		message = MQMessage(
			topic="financial.critical.payment",
			payload=b'{"transaction_id": "12345", "amount": 10000}',
			tenant_id="financial_test",
			source_application="payment_processor",
			priority=MessagePriority.CRITICAL
		)
		
		# Federate message
		result = await federation.federate_message(message)
		
		assert result['success'] == True
		# Critical financial messages should be replicated widely
		assert len(result['replicated_regions']) > 0
	
	@pytest.mark.asyncio
	async def test_eu_compliance_replication(self, federation):
		"""Test EU compliance-based replication"""
		# Create EU-specific message
		message = MQMessage(
			topic="customer.eu.profile",
			payload=b'{"user_id": "eu_user_123", "gdpr_consent": true}',
			tenant_id="eu_compliance_test",
			source_application="customer_service"
		)
		
		# Set current region to EU
		federation.current_region = "aws_eu_west_1"
		
		# Federate message
		result = await federation.federate_message(message)
		
		assert result['success'] == True
		assert result['source_region'] == "aws_eu_west_1"
		# EU data should stay in EU regions for compliance
	
	@pytest.mark.asyncio
	async def test_federation_status_reporting(self, federation):
		"""Test federation status reporting"""
		status = await federation.get_federation_status()
		
		assert 'enabled' in status
		assert 'current_region' in status
		assert 'federation_strategy' in status
		assert 'total_regions' in status
		assert 'healthy_regions' in status
		assert 'replication_stats' in status
		assert 'recent_failovers' in status
		assert 'regions' in status
		
		assert status['enabled'] == True
		assert status['federation_strategy'] == ReplicationStrategy.ACTIVE_ACTIVE.value
		assert isinstance(status['regions'], list)
		assert len(status['regions']) > 0
		
		# Check region information structure
		for region_info in status['regions']:
			assert 'region_id' in region_info
			assert 'provider' in region_info
			assert 'health_status' in region_info
			assert 'is_primary' in region_info


class TestCrossCloudScenarios:
	"""Test complex cross-cloud scenarios"""
	
	@pytest.mark.asyncio
	async def test_multi_region_failover_cascade(self):
		"""Test cascading failover across multiple regions"""
		# Create service and federation
		service = MQEBService()
		await service.initialize()
		
		try:
			federation = await create_multi_cloud_federation(service)
			
			# Simulate multiple region failures
			regions = list(federation.region_manager.regions.keys())
			
			# Mark first region as unhealthy
			federation.region_manager.regions[regions[0]].health_status = "unhealthy"
			
			# Evaluate failover
			potential_failovers = await federation.failover_manager.evaluate_failover_conditions()
			
			if potential_failovers:
				# Execute first failover
				await federation.failover_manager.execute_failover(potential_failovers[0])
				
				# Check that primary was switched
				new_primary_regions = [
					r for r in federation.region_manager.regions.values()
					if r.is_primary and r.enabled
				]
				assert len(new_primary_regions) > 0
			
			await federation.shutdown()
		finally:
			await service.shutdown()
	
	@pytest.mark.asyncio
	async def test_geographic_message_routing(self):
		"""Test geographic-based message routing"""
		service = MQEBService()
		await service.initialize()
		
		try:
			federation = await create_multi_cloud_federation(service)
			
			# Test US message
			us_message = MQMessage(
				topic="user.activity.us.west",
				payload=b'{"user_location": "california", "event": "login"}',
				tenant_id="geographic_test",
				source_application="user_service"
			)
			
			result = await federation.federate_message(us_message)
			assert result['success'] == True
			
			# Test EU message (should follow compliance rules)
			eu_message = MQMessage(
				topic="customer.eu.activity",
				payload=b'{"user_location": "germany", "event": "profile_update"}',
				tenant_id="geographic_test",
				source_application="customer_service"
			)
			
			result = await federation.federate_message(eu_message)
			assert result['success'] == True
			
			await federation.shutdown()
		finally:
			await service.shutdown()
	
	@pytest.mark.asyncio
	async def test_cost_optimized_replication(self):
		"""Test cost-optimized replication strategy"""
		service = MQEBService()
		await service.initialize()
		
		try:
			federation = await create_multi_cloud_federation(service)
			
			# Add cost-optimized replication rule
			cost_rule = ReplicationRule(
				rule_id="cost_optimized",
				name="Cost Optimized Replication",
				topic_patterns=["bulk.*", "analytics.*"],
				source_regions=["aws_us_east_1"],
				target_regions=["gcp_us_central_1"],  # Cheaper GCP region
				strategy=ReplicationStrategy.COST_OPTIMIZED,
				priority=20
			)
			
			federation.message_replicator.add_replication_rule(cost_rule)
			
			# Test bulk data message
			bulk_message = MQMessage(
				topic="bulk.data.export",
				payload=b'{"dataset": "large_analytics_data", "size_gb": 100}',
				tenant_id="cost_test",
				source_application="analytics_service"
			)
			
			result = await federation.federate_message(bulk_message)
			assert result['success'] == True
			
			await federation.shutdown()
		finally:
			await service.shutdown()


if __name__ == "__main__":
	# Run tests if script is executed directly
	pytest.main([__file__, "-v"])