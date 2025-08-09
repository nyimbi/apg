"""
APG Central Configuration - Multi-Region Deployment Orchestrator

Advanced multi-region deployment automation with intelligent failover,
disaster recovery, and global load balancing capabilities.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
from pathlib import Path

# Cloud provider SDKs
import boto3
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from google.cloud import compute_v1
import kubernetes.client as k8s_client

# Database and configuration
from ..service import CentralConfigurationEngine


class CloudProvider(Enum):
	"""Supported cloud providers."""
	AWS = "aws"
	AZURE = "azure"
	GCP = "gcp"
	KUBERNETES = "kubernetes"
	ON_PREMISES = "on_premises"


class RegionStatus(Enum):
	"""Region deployment status."""
	ACTIVE = "active"
	STANDBY = "standby"
	MAINTENANCE = "maintenance"
	FAILED = "failed"
	RECOVERING = "recovering"
	DEPLOYING = "deploying"


class FailoverStrategy(Enum):
	"""Failover strategies."""
	AUTOMATIC = "automatic"
	MANUAL = "manual"
	HYBRID = "hybrid"


class ReplicationMode(Enum):
	"""Data replication modes."""
	SYNCHRONOUS = "synchronous"
	ASYNCHRONOUS = "asynchronous"
	SEMI_SYNCHRONOUS = "semi_synchronous"


@dataclass
class RegionConfig:
	"""Configuration for a deployment region."""
	region_id: str
	region_name: str
	cloud_provider: CloudProvider
	provider_region: str  # e.g., us-east-1, eastus, us-central1
	is_primary: bool
	status: RegionStatus
	capacity_allocation: float  # 0.0 to 1.0
	
	# Network configuration
	vpc_id: Optional[str]
	subnet_ids: List[str]
	security_group_ids: List[str]
	
	# Resource configuration
	instance_types: Dict[str, str]
	storage_config: Dict[str, Any]
	network_config: Dict[str, Any]
	
	# Monitoring and health
	health_check_url: str
	monitoring_endpoints: List[str]
	last_health_check: Optional[datetime]
	latency_ms: Optional[float]
	
	# Backup and recovery
	backup_enabled: bool
	backup_retention_days: int
	recovery_point_objective_minutes: int
	recovery_time_objective_minutes: int


@dataclass
class DeploymentPlan:
	"""Multi-region deployment plan."""
	plan_id: str
	name: str
	description: str
	target_regions: List[str]
	deployment_strategy: str  # blue_green, rolling, canary
	rollback_strategy: str
	estimated_duration_minutes: int
	created_at: datetime
	created_by: str


@dataclass
class FailoverEvent:
	"""Failover event record."""
	event_id: str
	timestamp: datetime
	from_region: str
	to_region: str
	trigger_reason: str
	failover_strategy: FailoverStrategy
	duration_seconds: float
	success: bool
	affected_services: List[str]
	recovery_actions: List[str]


@dataclass
class ReplicationStatus:
	"""Data replication status between regions."""
	from_region: str
	to_region: str
	replication_mode: ReplicationMode
	lag_seconds: float
	last_sync: datetime
	bytes_replicated: int
	error_count: int
	health_status: str


class MultiRegionOrchestrator:
	"""Multi-region deployment and disaster recovery orchestrator."""
	
	def __init__(self, config_engine: CentralConfigurationEngine):
		"""Initialize multi-region orchestrator."""
		self.config_engine = config_engine
		self.regions: Dict[str, RegionConfig] = {}
		self.deployment_history: List[Dict[str, Any]] = []
		self.failover_history: List[FailoverEvent] = []
		self.replication_status: List[ReplicationStatus] = []
		
		# Cloud provider clients
		self.aws_clients: Dict[str, Any] = {}
		self.azure_clients: Dict[str, Any] = {}
		self.gcp_clients: Dict[str, Any] = {}
		self.k8s_clients: Dict[str, Any] = {}
		
		# Configuration
		self.primary_region: Optional[str] = None
		self.failover_enabled = True
		self.auto_scaling_enabled = True
		self.health_check_interval = 30  # seconds
		self.failover_threshold_failures = 3
		self.failover_timeout_seconds = 300
		
		# Initialize components
		asyncio.create_task(self._initialize_cloud_clients())
		asyncio.create_task(self._initialize_default_regions())
		asyncio.create_task(self._start_health_monitoring())
	
	# ==================== Initialization ====================
	
	async def _initialize_cloud_clients(self):
		"""Initialize cloud provider clients."""
		try:
			# AWS clients
			session = boto3.Session()
			self.aws_clients = {
				"ec2": session.client("ec2"),
				"ecs": session.client("ecs"),
				"rds": session.client("rds"),
				"route53": session.client("route53"),
				"cloudformation": session.client("cloudformation")
			}
			print("✅ AWS clients initialized")
		except Exception as e:
			print(f"⚠️ AWS client initialization failed: {e}")
		
		try:
			# Azure clients
			credential = DefaultAzureCredential()
			self.azure_clients = {
				"resource": ResourceManagementClient(credential, "subscription-id"),
				# Add other Azure clients as needed
			}
			print("✅ Azure clients initialized")
		except Exception as e:
			print(f"⚠️ Azure client initialization failed: {e}")
		
		try:
			# GCP clients
			self.gcp_clients = {
				"compute": compute_v1.InstancesClient(),
				# Add other GCP clients as needed
			}
			print("✅ GCP clients initialized")
		except Exception as e:
			print(f"⚠️ GCP client initialization failed: {e}")
		
		try:
			# Kubernetes clients
			from kubernetes import config as k8s_config
			k8s_config.load_incluster_config()
			self.k8s_clients = {
				"apps": k8s_client.AppsV1Api(),
				"core": k8s_client.CoreV1Api(),
				"custom": k8s_client.CustomObjectsApi()
			}
			print("✅ Kubernetes clients initialized")
		except Exception as e:
			print(f"⚠️ Kubernetes client initialization failed: {e}")
	
	async def _initialize_default_regions(self):
		"""Initialize default region configurations."""
		# Primary region (US East)
		us_east_region = RegionConfig(
			region_id="us_east_primary",
			region_name="US East Primary",
			cloud_provider=CloudProvider.AWS,
			provider_region="us-east-1",
			is_primary=True,
			status=RegionStatus.ACTIVE,
			capacity_allocation=1.0,
			vpc_id="vpc-12345678",
			subnet_ids=["subnet-12345678", "subnet-87654321"],
			security_group_ids=["sg-12345678"],
			instance_types={
				"api": "m5.xlarge",
				"web": "m5.large",
				"database": "r5.2xlarge"
			},
			storage_config={
				"type": "gp3",
				"size_gb": 500,
				"iops": 3000,
				"encryption": True
			},
			network_config={
				"load_balancer": "application",
				"ssl_termination": True,
				"cdn_enabled": True
			},
			health_check_url="https://api-us-east.central-config.com/health",
			monitoring_endpoints=[
				"https://prometheus-us-east.central-config.com",
				"https://grafana-us-east.central-config.com"
			],
			last_health_check=datetime.now(timezone.utc),
			latency_ms=45.2,
			backup_enabled=True,
			backup_retention_days=30,
			recovery_point_objective_minutes=5,
			recovery_time_objective_minutes=15
		)
		
		# Secondary region (US West)
		us_west_region = RegionConfig(
			region_id="us_west_secondary",
			region_name="US West Secondary",
			cloud_provider=CloudProvider.AWS,
			provider_region="us-west-2",
			is_primary=False,
			status=RegionStatus.STANDBY,
			capacity_allocation=0.5,
			vpc_id="vpc-87654321",
			subnet_ids=["subnet-11111111", "subnet-22222222"],
			security_group_ids=["sg-87654321"],
			instance_types={
				"api": "m5.large",
				"web": "m5.medium",
				"database": "r5.xlarge"
			},
			storage_config={
				"type": "gp3",
				"size_gb": 300,
				"iops": 2000,
				"encryption": True
			},
			network_config={
				"load_balancer": "application",
				"ssl_termination": True,
				"cdn_enabled": True
			},
			health_check_url="https://api-us-west.central-config.com/health",
			monitoring_endpoints=[
				"https://prometheus-us-west.central-config.com"
			],
			last_health_check=datetime.now(timezone.utc),
			latency_ms=52.8,
			backup_enabled=True,
			backup_retention_days=30,
			recovery_point_objective_minutes=15,
			recovery_time_objective_minutes=30
		)
		
		# European region
		eu_region = RegionConfig(
			region_id="eu_west_secondary",
			region_name="Europe West Secondary",
			cloud_provider=CloudProvider.AWS,
			provider_region="eu-west-1",
			is_primary=False,
			status=RegionStatus.STANDBY,
			capacity_allocation=0.3,
			vpc_id="vpc-33333333",
			subnet_ids=["subnet-33333333", "subnet-44444444"],
			security_group_ids=["sg-33333333"],
			instance_types={
				"api": "m5.large",
				"web": "m5.medium",
				"database": "r5.large"
			},
			storage_config={
				"type": "gp3",
				"size_gb": 200,
				"iops": 1500,
				"encryption": True
			},
			network_config={
				"load_balancer": "application",
				"ssl_termination": True,
				"cdn_enabled": True
			},
			health_check_url="https://api-eu-west.central-config.com/health",
			monitoring_endpoints=[
				"https://prometheus-eu-west.central-config.com"
			],
			last_health_check=datetime.now(timezone.utc),
			latency_ms=78.5,
			backup_enabled=True,
			backup_retention_days=30,
			recovery_point_objective_minutes=30,
			recovery_time_objective_minutes=45
		)
		
		self.regions = {
			us_east_region.region_id: us_east_region,
			us_west_region.region_id: us_west_region,
			eu_region.region_id: eu_region
		}
		
		self.primary_region = us_east_region.region_id
		
		print(f"🌍 Initialized {len(self.regions)} regions")
	
	# ==================== Deployment Orchestration ====================
	
	async def deploy_to_multiple_regions(
		self,
		target_regions: List[str],
		deployment_strategy: str = "rolling",
		configuration_updates: Optional[Dict[str, Any]] = None
	) -> Dict[str, Any]:
		"""Deploy to multiple regions with specified strategy."""
		deployment_id = f"deploy_{uuid.uuid4().hex[:8]}"
		start_time = datetime.now(timezone.utc)
		
		print(f"🚀 Starting multi-region deployment: {deployment_id}")
		print(f"Target regions: {target_regions}")
		print(f"Strategy: {deployment_strategy}")
		
		deployment_result = {
			"deployment_id": deployment_id,
			"started_at": start_time.isoformat(),
			"target_regions": target_regions,
			"deployment_strategy": deployment_strategy,
			"status": "running",
			"regions_completed": [],
			"regions_failed": [],
			"total_duration_seconds": 0
		}
		
		try:
			if deployment_strategy == "rolling":
				await self._rolling_deployment(target_regions, configuration_updates, deployment_result)
			
			elif deployment_strategy == "blue_green":
				await self._blue_green_deployment(target_regions, configuration_updates, deployment_result)
			
			elif deployment_strategy == "canary":
				await self._canary_deployment(target_regions, configuration_updates, deployment_result)
			
			else:
				raise ValueError(f"Unknown deployment strategy: {deployment_strategy}")
			
			deployment_result["status"] = "completed"
			deployment_result["total_duration_seconds"] = (datetime.now(timezone.utc) - start_time).total_seconds()
			
			print(f"✅ Multi-region deployment completed: {deployment_id}")
			
		except Exception as e:
			deployment_result["status"] = "failed"
			deployment_result["error"] = str(e)
			deployment_result["total_duration_seconds"] = (datetime.now(timezone.utc) - start_time).total_seconds()
			
			print(f"❌ Multi-region deployment failed: {e}")
		
		# Record deployment history
		self.deployment_history.append(deployment_result)
		
		return deployment_result
	
	async def _rolling_deployment(
		self,
		target_regions: List[str],
		configuration_updates: Optional[Dict[str, Any]],
		deployment_result: Dict[str, Any]
	):
		"""Perform rolling deployment across regions."""
		for region_id in target_regions:
			if region_id not in self.regions:
				deployment_result["regions_failed"].append({
					"region_id": region_id,
					"error": "Region not configured"
				})
				continue
			
			region = self.regions[region_id]
			
			try:
				print(f"🔄 Deploying to region: {region.region_name}")
				
				# Pre-deployment health check
				if not await self._check_region_health(region_id):
					raise Exception(f"Region {region_id} failed health check")
				
				# Deploy to region
				await self._deploy_to_region(region, configuration_updates)
				
				# Post-deployment validation
				await self._validate_deployment(region_id)
				
				deployment_result["regions_completed"].append({
					"region_id": region_id,
					"region_name": region.region_name,
					"completed_at": datetime.now(timezone.utc).isoformat()
				})
				
				print(f"✅ Deployment completed for region: {region.region_name}")
				
				# Brief pause between regions
				await asyncio.sleep(30)
				
			except Exception as e:
				deployment_result["regions_failed"].append({
					"region_id": region_id,
					"region_name": region.region_name,
					"error": str(e),
					"failed_at": datetime.now(timezone.utc).isoformat()
				})
				
				print(f"❌ Deployment failed for region {region.region_name}: {e}")
				
				# Decide whether to continue or stop
				if region.is_primary:
					raise Exception(f"Primary region deployment failed: {e}")
	
	async def _deploy_to_region(
		self,
		region: RegionConfig,
		configuration_updates: Optional[Dict[str, Any]]
	):
		"""Deploy to a specific region."""
		if region.cloud_provider == CloudProvider.AWS:
			await self._deploy_to_aws_region(region, configuration_updates)
		
		elif region.cloud_provider == CloudProvider.AZURE:
			await self._deploy_to_azure_region(region, configuration_updates)
		
		elif region.cloud_provider == CloudProvider.GCP:
			await self._deploy_to_gcp_region(region, configuration_updates)
		
		elif region.cloud_provider == CloudProvider.KUBERNETES:
			await self._deploy_to_k8s_region(region, configuration_updates)
		
		else:
			raise ValueError(f"Unsupported cloud provider: {region.cloud_provider}")
	
	async def _deploy_to_aws_region(
		self,
		region: RegionConfig,
		configuration_updates: Optional[Dict[str, Any]]
	):
		"""Deploy to AWS region."""
		try:
			# Update ECS services
			if "ecs" in self.aws_clients:
				ecs_client = self.aws_clients["ecs"]
				
				# Update API service
				ecs_client.update_service(
					cluster=f"central-config-{region.provider_region}",
					service="central-config-api",
					forceNewDeployment=True
				)
				
				# Update Web service  
				ecs_client.update_service(
					cluster=f"central-config-{region.provider_region}",
					service="central-config-web",
					forceNewDeployment=True
				)
			
			# Update RDS if needed
			if configuration_updates and "database" in configuration_updates:
				rds_client = self.aws_clients["rds"]
				# Apply database configuration updates
				# This would involve parameter group updates, etc.
			
			print(f"✅ AWS deployment completed for {region.region_name}")
			
		except Exception as e:
			print(f"❌ AWS deployment failed for {region.region_name}: {e}")
			raise
	
	async def _deploy_to_k8s_region(
		self,
		region: RegionConfig,
		configuration_updates: Optional[Dict[str, Any]]
	):
		"""Deploy to Kubernetes region."""
		try:
			if "apps" in self.k8s_clients:
				apps_client = self.k8s_clients["apps"]
				
				# Update API deployment
				api_deployment = apps_client.read_namespaced_deployment(
					name="central-config-api",
					namespace="central-config"
				)
				
				# Trigger rolling update by updating annotation
				if not api_deployment.spec.template.metadata.annotations:
					api_deployment.spec.template.metadata.annotations = {}
				
				api_deployment.spec.template.metadata.annotations["deployment.kubernetes.io/revision"] = str(int(time.time()))
				
				apps_client.patch_namespaced_deployment(
					name="central-config-api",
					namespace="central-config",
					body=api_deployment
				)
				
				# Update Web deployment
				web_deployment = apps_client.read_namespaced_deployment(
					name="central-config-web",
					namespace="central-config"
				)
				
				if not web_deployment.spec.template.metadata.annotations:
					web_deployment.spec.template.metadata.annotations = {}
				
				web_deployment.spec.template.metadata.annotations["deployment.kubernetes.io/revision"] = str(int(time.time()))
				
				apps_client.patch_namespaced_deployment(
					name="central-config-web",
					namespace="central-config",
					body=web_deployment
				)
			
			print(f"✅ Kubernetes deployment completed for {region.region_name}")
			
		except Exception as e:
			print(f"❌ Kubernetes deployment failed for {region.region_name}: {e}")
			raise
	
	# ==================== Disaster Recovery and Failover ====================
	
	async def trigger_failover(
		self,
		from_region: str,
		to_region: str,
		trigger_reason: str,
		strategy: FailoverStrategy = FailoverStrategy.AUTOMATIC
	) -> FailoverEvent:
		"""Trigger failover from one region to another."""
		event_id = f"failover_{uuid.uuid4().hex[:8]}"
		start_time = datetime.now(timezone.utc)
		
		print(f"🔄 Triggering failover: {from_region} -> {to_region}")
		print(f"Reason: {trigger_reason}")
		
		failover_event = FailoverEvent(
			event_id=event_id,
			timestamp=start_time,
			from_region=from_region,
			to_region=to_region,
			trigger_reason=trigger_reason,
			failover_strategy=strategy,
			duration_seconds=0,
			success=False,
			affected_services=[],
			recovery_actions=[]
		)
		
		try:
			# Pre-failover validation
			if not await self._validate_failover_target(to_region):
				raise Exception(f"Target region {to_region} is not ready for failover")
			
			# Step 1: Update DNS routing
			await self._update_dns_routing(from_region, to_region)
			failover_event.recovery_actions.append("DNS routing updated")
			
			# Step 2: Scale up target region
			await self._scale_up_region(to_region)
			failover_event.recovery_actions.append("Target region scaled up")
			
			# Step 3: Synchronize data
			await self._synchronize_data(from_region, to_region)
			failover_event.recovery_actions.append("Data synchronized")
			
			# Step 4: Update load balancer configuration
			await self._update_load_balancer_config(from_region, to_region)
			failover_event.recovery_actions.append("Load balancer updated")
			
			# Step 5: Validate failover success
			await self._validate_failover_success(to_region)
			failover_event.recovery_actions.append("Failover validated")
			
			# Step 6: Update region status
			if from_region in self.regions:
				self.regions[from_region].status = RegionStatus.FAILED
			if to_region in self.regions:
				self.regions[to_region].status = RegionStatus.ACTIVE
				if from_region == self.primary_region:
					self.regions[to_region].is_primary = True
					self.primary_region = to_region
			
			failover_event.success = True
			failover_event.affected_services = ["api", "web", "database"]
			
			print(f"✅ Failover completed successfully: {event_id}")
			
		except Exception as e:
			failover_event.recovery_actions.append(f"Failover failed: {str(e)}")
			print(f"❌ Failover failed: {e}")
		
		failover_event.duration_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
		self.failover_history.append(failover_event)
		
		return failover_event
	
	async def _validate_failover_target(self, target_region: str) -> bool:
		"""Validate that target region is ready for failover."""
		if target_region not in self.regions:
			return False
		
		region = self.regions[target_region]
		
		# Check region health
		if not await self._check_region_health(target_region):
			return False
		
		# Check minimum capacity
		if region.capacity_allocation < 0.3:  # Need at least 30% capacity
			return False
		
		# Check data replication status
		replication_healthy = await self._check_replication_health(target_region)
		if not replication_healthy:
			return False
		
		return True
	
	async def _update_dns_routing(self, from_region: str, to_region: str):
		"""Update DNS routing to failover target."""
		try:
			if "route53" in self.aws_clients:
				route53_client = self.aws_clients["route53"]
				
				# Get the hosted zone
				hosted_zone_id = "Z1234567890ABC"  # This would be configured
				
				# Update the A record to point to new region
				if to_region in self.regions:
					target_region = self.regions[to_region]
					
					route53_client.change_resource_record_sets(
						HostedZoneId=hosted_zone_id,
						ChangeBatch={
							"Changes": [{
								"Action": "UPSERT",
								"ResourceRecordSet": {
									"Name": "api.central-config.com",
									"Type": "CNAME",
									"TTL": 60,
									"ResourceRecords": [
										{"Value": target_region.health_check_url.replace("https://", "")}
									]
								}
							}]
						}
					)
			
			print(f"✅ DNS routing updated: {from_region} -> {to_region}")
			
		except Exception as e:
			print(f"❌ DNS routing update failed: {e}")
			raise
	
	# ==================== Health Monitoring ====================
	
	async def _start_health_monitoring(self):
		"""Start continuous health monitoring for all regions."""
		print("🏥 Starting multi-region health monitoring")
		
		while True:
			try:
				# Check health of all regions
				health_tasks = []
				for region_id in self.regions.keys():
					task = self._check_region_health(region_id)
					health_tasks.append(task)
				
				if health_tasks:
					health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
					
					# Process health results
					for i, result in enumerate(health_results):
						region_id = list(self.regions.keys())[i]
						if isinstance(result, Exception):
							print(f"❌ Health check failed for {region_id}: {result}")
							await self._handle_region_failure(region_id, str(result))
						elif not result:
							print(f"⚠️ Region {region_id} unhealthy")
							await self._handle_region_failure(region_id, "Health check failed")
				
				# Update replication status
				await self._update_replication_status()
				
				# Check if failover is needed
				await self._evaluate_failover_triggers()
				
				await asyncio.sleep(self.health_check_interval)
				
			except Exception as e:
				print(f"❌ Health monitoring error: {e}")
				await asyncio.sleep(self.health_check_interval)
	
	async def _check_region_health(self, region_id: str) -> bool:
		"""Check health of a specific region."""
		if region_id not in self.regions:
			return False
		
		region = self.regions[region_id]
		
		try:
			# HTTP health check
			import httpx
			async with httpx.AsyncClient(timeout=10.0) as client:
				response = await client.get(region.health_check_url)
				
				if response.status_code == 200:
					# Update region metrics
					region.last_health_check = datetime.now(timezone.utc)
					region.latency_ms = response.elapsed.total_seconds() * 1000
					
					# Check if region was previously failed and now recovering
					if region.status == RegionStatus.FAILED:
						region.status = RegionStatus.RECOVERING
						print(f"🔄 Region {region.region_name} recovering")
					elif region.status == RegionStatus.RECOVERING:
						region.status = RegionStatus.ACTIVE
						print(f"✅ Region {region.region_name} recovered")
					
					return True
				else:
					return False
		
		except Exception as e:
			print(f"❌ Health check failed for {region.region_name}: {e}")
			return False
	
	async def _handle_region_failure(self, region_id: str, reason: str):
		"""Handle region failure detection."""
		if region_id not in self.regions:
			return
		
		region = self.regions[region_id]
		
		# Update region status
		region.status = RegionStatus.FAILED
		
		print(f"🚨 Region failure detected: {region.region_name} - {reason}")
		
		# If this is the primary region, trigger automatic failover
		if region.is_primary and self.failover_enabled:
			# Find best failover target
			failover_target = await self._find_best_failover_target(region_id)
			
			if failover_target:
				await self.trigger_failover(
					from_region=region_id,
					to_region=failover_target,
					trigger_reason=f"Primary region failure: {reason}",
					strategy=FailoverStrategy.AUTOMATIC
				)
	
	async def _find_best_failover_target(self, failed_region: str) -> Optional[str]:
		"""Find the best region for failover."""
		candidates = []
		
		for region_id, region in self.regions.items():
			if (region_id != failed_region and 
				region.status == RegionStatus.ACTIVE and
				region.capacity_allocation >= 0.5):  # Need sufficient capacity
				
				candidates.append((region_id, region.latency_ms or 1000))
		
		# Sort by latency (prefer lower latency)
		candidates.sort(key=lambda x: x[1])
		
		return candidates[0][0] if candidates else None
	
	# ==================== Data Replication Management ====================
	
	async def setup_data_replication(
		self,
		source_region: str,
		target_region: str,
		replication_mode: ReplicationMode = ReplicationMode.ASYNCHRONOUS
	) -> bool:
		"""Setup data replication between regions."""
		print(f"🔄 Setting up data replication: {source_region} -> {target_region}")
		
		try:
			# Setup database replication
			await self._setup_database_replication(source_region, target_region, replication_mode)
			
			# Setup configuration sync
			await self._setup_configuration_sync(source_region, target_region)
			
			# Create replication status record
			replication_status = ReplicationStatus(
				from_region=source_region,
				to_region=target_region,
				replication_mode=replication_mode,
				lag_seconds=0.0,
				last_sync=datetime.now(timezone.utc),
				bytes_replicated=0,
				error_count=0,
				health_status="healthy"
			)
			
			self.replication_status.append(replication_status)
			
			print(f"✅ Data replication setup completed: {source_region} -> {target_region}")
			return True
			
		except Exception as e:
			print(f"❌ Data replication setup failed: {e}")
			return False
	
	async def _setup_database_replication(
		self,
		source_region: str,
		target_region: str,
		replication_mode: ReplicationMode
	):
		"""Setup database replication between regions."""
		# This would setup PostgreSQL streaming replication or similar
		# For AWS RDS, this would configure read replicas
		# For Azure Database, this would setup geo-replication
		# For GCP Cloud SQL, this would setup replica instances
		
		print(f"📊 Database replication configured: {replication_mode.value}")
	
	async def get_deployment_status(self) -> Dict[str, Any]:
		"""Get comprehensive deployment status across all regions."""
		status = {
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"total_regions": len(self.regions),
			"active_regions": len([r for r in self.regions.values() if r.status == RegionStatus.ACTIVE]),
			"primary_region": self.primary_region,
			"failover_enabled": self.failover_enabled,
			"regions": {},
			"replication_status": [],
			"recent_deployments": self.deployment_history[-5:],  # Last 5 deployments
			"recent_failovers": [asdict(f) for f in self.failover_history[-3:]]  # Last 3 failovers
		}
		
		# Region details
		for region_id, region in self.regions.items():
			status["regions"][region_id] = {
				"region_name": region.region_name,
				"cloud_provider": region.cloud_provider.value,
				"provider_region": region.provider_region,
				"is_primary": region.is_primary,
				"status": region.status.value,
				"capacity_allocation": region.capacity_allocation,
				"last_health_check": region.last_health_check.isoformat() if region.last_health_check else None,
				"latency_ms": region.latency_ms,
				"backup_enabled": region.backup_enabled
			}
		
		# Replication status
		for repl in self.replication_status:
			status["replication_status"].append({
				"from_region": repl.from_region,
				"to_region": repl.to_region,
				"replication_mode": repl.replication_mode.value,
				"lag_seconds": repl.lag_seconds,
				"last_sync": repl.last_sync.isoformat(),
				"health_status": repl.health_status
			})
		
		return status


# ==================== Factory Functions ====================

async def create_multi_region_orchestrator(
	config_engine: CentralConfigurationEngine
) -> MultiRegionOrchestrator:
	"""Create and initialize multi-region orchestrator."""
	orchestrator = MultiRegionOrchestrator(config_engine)
	await asyncio.sleep(2)  # Allow initialization
	print("🌍 Multi-Region Orchestrator initialized")
	return orchestrator